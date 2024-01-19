/*
Copyright 2023 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cuda_efficient_features.h"

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "device_buffer.h"
#include "cuda_efficient_features_internal.h"
#include "cuda_efficient_descriptors.h"

namespace cv
{
namespace cuda
{

static constexpr int PATCH_SIZE = 31;
static constexpr int HALF_PATCH_SIZE = 15;
static constexpr double CORNER_DENSITY = 0.1;

void calcKeypoints(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, int nfeatures, int threshold,
	GpuMat& d_buffer, HostMem& h_buffer, cudaStream_t stream);
int radiusSuppressionBufferSize(Size imgSize, int npoints);
void radiusSuppression(const GpuMat& src, GpuMat& dst, Size imgSize, float radius,
	GpuMat& d_buffer, HostMem& h_buffer, cudaStream_t stream);
void limitPoints(GpuMat& points, int maxpoints, cudaStream_t stream);
void calcResponses(const GpuMat& image, GpuMat& points, cudaStream_t stream);
void calcAngles(const GpuMat& image, GpuMat& points, cudaStream_t stream);
void scalePoints(GpuMat& points, float scale, int octave, cudaStream_t stream);
void convertKeypoints(const GpuMat& src, GpuMat& dst, cudaStream_t stream);

static Ptr<EfficientDescriptorsAsync> createDescriber(EfficientFeatures::DescriptorType descriptorType)
{
	switch (descriptorType)
	{
	case EfficientFeatures::BAD_256:
		return cuda::BAD::create(1, cuda::BAD::SIZE_256_BITS);
		break;
	case EfficientFeatures::BAD_512:
		return cuda::BAD::create(1, cuda::BAD::SIZE_512_BITS);
		break;
	case EfficientFeatures::HASH_SIFT_256:
		return cuda::HashSIFT::create(1, cuda::HashSIFT::SIZE_256_BITS);
		break;
	case EfficientFeatures::HASH_SIFT_512:
		return cuda::HashSIFT::create(1, cuda::HashSIFT::SIZE_512_BITS);
		break;
	default:
		return nullptr;
	}

	return nullptr;
}

void getInputMat(InputArray src, GpuMat& dst, Stream& stream)
{
	switch (src.kind())
	{
	case _InputArray::KindFlag::MAT:
		dst.upload(src, stream);
		break;
	case _InputArray::KindFlag::CUDA_GPU_MAT:
		dst = src.getGpuMat();
		break;
	default:
		CV_Error(Error::StsBadArg, "Unsupported");
	}
}

void getOutputMat(OutputArray src, GpuMat& dst, int rows, int cols, int type)
{
	switch (src.kind())
	{
	case _InputArray::KindFlag::MAT:
		dst.create(rows, cols, type);
		break;
	case _InputArray::KindFlag::CUDA_GPU_MAT:
		src.create(rows, cols, type);
		dst = src.getGpuMat();
		break;
	default:
		CV_Error(Error::StsBadArg, "Unsupported");
	}
}

void getKeypointsMat(InputKeyPoints _src, GpuMat& dst, Stream& stream)
{
	if (std::holds_alternative<_InputArray>(_src))
	{
		InputArray src = std::get<_InputArray>(_src);

		GpuMat tmp;
		getInputMat(src, tmp, stream);

		CV_Assert(tmp.rows == 5 && tmp.type() == CV_32F);

		dst.create(tmp.cols, 1, CV_32FC4);
		convertKeypoints(tmp, dst, StreamAccessor::getStream(stream));
	}
	else
	{
		const KeyPoints& src = std::get<KeyPoints>(_src);

		const int nkeypoints = static_cast<int>(src.size());
		Mat tmp(nkeypoints, 1, CV_32FC4);
		for (int i = 0; i < nkeypoints; i++)
		{
			const auto& kpt = src[i];
			tmp.at<cv::Vec4f>(i) = cv::Vec4f(kpt.pt.x, kpt.pt.y, kpt.size, kpt.angle);
		}
		dst.upload(tmp, stream);
	}
}

bool isEmpty(InputKeyPoints keypoints)
{
	return std::visit([](const auto& v) { return v.empty(); }, keypoints);
}

static void calcImagePyramid(const GpuMat& image, std::vector<GpuMat>& images, std::vector<float>& scales,
	float scaleFactor, int nlevels, Stream& stream)
{
	CV_Assert(image.type() == CV_8U);

	images.resize(nlevels);
	scales.resize(nlevels);

	float scale = 1.f;
	image.copyTo(images[0]);
	scales[0] = scale;

	for (int s = 1; s < nlevels; s++)
	{
		scale *= scaleFactor;
		const float invScale = 1.f / scale;
		const int h = cvRound(invScale * image.rows);
		const int w = cvRound(invScale * image.cols);
		resize(images[s - 1], images[s], Size(w, h), 0, 0, INTER_LINEAR, stream);
		scales[s] = scale;
	}
}

static void calcNumFeaturesPerLevel(int total, float scaleFactor, int nlevels, std::vector<int>& nfeaturesPerLevel)
{
	// compute number of features in each scale
	nfeaturesPerLevel.resize(nlevels);

	const double factor = 1 / scaleFactor;
	double nfeatues = total * (1 - factor) / (1 - std::pow(factor, nlevels));
	int sumfeatures = 0;
	for (int s = 0; s < nlevels - 1; s++)
	{
		nfeaturesPerLevel[s] = cvRound(nfeatues);
		sumfeatures += nfeaturesPerLevel[s];
		nfeatues *= factor;
	}
	nfeaturesPerLevel[nlevels - 1] = std::max(total - sumfeatures, 0);
}

static void createMask(GpuMat& mask, Size imgSize, int border, Stream& stream)
{
	mask.create(imgSize, CV_8U);
	mask.setTo(Scalar::all(0), stream);
	const Rect ROI(border, border, imgSize.width - 2 * border, imgSize.height - 2 * border);
	mask(ROI).setTo(Scalar::all(255), stream);
}

static inline float convertToDegree(float angle)
{
	constexpr float PI = static_cast<float>(CV_PI);
	if (angle < 0)
		angle += 2.f * PI;
	return (180.f / PI) * angle;
}

class EfficientFeaturesImpl : public EfficientFeatures
{
public:

	EfficientFeaturesImpl(int nfeatures, float scaleFactor, int nlevels,
		int firstLevel, int fastThreshold, int nonmaxRadius, DescriptorType descriptorType) : nfeatures_(nfeatures), scaleFactor_(scaleFactor),
		nlevels_(nlevels), firstLevel_(firstLevel), fastThreshold_(fastThreshold), nonmaxRadius_(nonmaxRadius), descriptorType_(descriptorType)
	{
		describer_ = createDescriber(descriptorType);
		filter_ = cuda::createGaussianFilter(CV_8UC1, -1, Size(7, 7), 2, 2, BORDER_REFLECT_101);
		h_buffer_.create(1, 16, CV_32S);
	}

	void detect(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) override
	{
		detectAsync(image, keypoints_, mask, Stream::Null());
		convert(keypoints_, keypoints);
	}

	void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) override
	{
		describer_->compute(image, keypoints, descriptors);
	}

	void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints, OutputArray descriptors,
		bool useProvidedKeypoints) override
	{
		detectAndComputeAsync(image, mask, keypoints_, descriptors, useProvidedKeypoints, Stream::Null());
		convert(keypoints_, keypoints);
	}

	void detectAsync(InputArray image, OutputArray keypoints, InputArray mask, Stream& stream) override
	{
		detectAndComputeAsync(image, mask, keypoints, noArray(), false, stream);
	}

	void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream) override
	{
		describer_->computeAsync(image, keypoints, descriptors, stream);
	}

	void detectAndComputeAsync(InputArray _image, InputArray _mask, OutputArray _keypoints, OutputArray _descriptors,
		bool useProvidedKeypoints, Stream& stream) override
	{
		CV_Assert(_image.type() == CV_8U);
		CV_Assert(!useProvidedKeypoints);

		const bool needDescriptors = _descriptors.needed();
		const cudaStream_t cuStream = StreamAccessor::getStream(stream);

		getInputMat(_image, image_, stream);

		calcImagePyramid(image_, imagePyr_, scales_, scaleFactor_, nlevels_, stream);

		calcNumFeaturesPerLevel(nfeatures_, scaleFactor_, nlevels_, nfeaturesPerLevel_);

		int nkeypoints = 0;
		maskPyr_.resize(nlevels_);
		kptsPyr_.resize(nlevels_);
		kptsBuf_.resize(nlevels_);
		for (int s = firstLevel_; s < nlevels_; s++)
		{
			const GpuMat& image = imagePyr_[s];
			GpuMat& mask = maskPyr_[s];

			if (mask.size() != image.size())
				createMask(mask, image.size(), HALF_PATCH_SIZE, stream);

			const int maxpoints = cvRound(CORNER_DENSITY * image.size().area());
			GpuMat tmppoints = fastBuf_.createMat(4, maxpoints, CV_32F);
			GpuMat keypoints = kptsBuf_[s].createMat(ROWS_COUNT, maxpoints, CV_32F);

			const int bufferSize = radiusSuppressionBufferSize(image.size(), maxpoints);
			GpuMat d_buffer = suppBuf_.createMat(bufferSize, 1, CV_32S);

			calcKeypoints(image, mask, tmppoints, maxpoints,
				fastThreshold_, d_buffer, h_buffer_, cuStream);

			calcResponses(image, tmppoints, cuStream);

			radiusSuppression(tmppoints, keypoints, image.size(), nonmaxRadius_,
				d_buffer, h_buffer_, cuStream);

			limitPoints(keypoints, nfeaturesPerLevel_[s], cuStream);

			calcAngles(image, keypoints, cuStream);

			kptsPyr_[s] = keypoints;
			nkeypoints += keypoints.cols;
		}

		if (nkeypoints == 0)
		{
			_keypoints.release();
			if (needDescriptors)
				descriptors_.release();
			return;
		}

		getOutputMat(_keypoints, keypoints_, ROWS_COUNT, nkeypoints, CV_32F);

		if (needDescriptors)
		{
			getOutputMat(_descriptors, descriptors_, nkeypoints, descriptorSize(), descriptorType());
			blurPyr_.resize(nlevels_);
			descPyr_.resize(nlevels_);
		}

		int offset = 0;
		for (int s = firstLevel_; s < nlevels_; s++)
		{
			GpuMat& keypoints = kptsPyr_[s];
			const int npoints = keypoints.cols;
			if (npoints == 0)
				continue;

			const Range dstRange(offset, offset + npoints);

			if (needDescriptors)
			{
				GpuMat descriptors = descriptors_.rowRange(dstRange);
				filter_->apply(imagePyr_[s], blurPyr_[s], stream);
				describer_->computeAsync(blurPyr_[s], keypoints, descriptors, stream);
			}

			// insert keypoints
			scalePoints(keypoints, scales_[s], s, cuStream);
			keypoints.copyTo(keypoints_.colRange(dstRange), stream);

			offset += npoints;
		}

		if (_keypoints.kind() == _InputArray::KindFlag::MAT)
			keypoints_.download(_keypoints, stream);

		if (needDescriptors && _descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors, stream);
	}

	void convert(InputArray src, CV_OUT std::vector<KeyPoint>& dst) override
	{
		Mat tmp;
		if (src.kind() == _InputArray::KindFlag::MAT)
			tmp = src.getMat();
		else if (src.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
			src.getGpuMat().download(tmp);

		const Vec2s* points = tmp.ptr<Vec2s>(LOCATION_ROW);
		const float* responses = tmp.ptr<float>(RESPONSE_ROW);
		const float* angles = tmp.ptr<float>(ANGLE_ROW);
		const int* octaves = tmp.ptr<int>(OCTAVE_ROW);
		const float* sizes = tmp.ptr<float>(SIZE_ROW);

		const int nkeypoints = tmp.cols;
		dst.resize(nkeypoints);
		for (int i = 0; i < nkeypoints; i++)
		{
			KeyPoint kpt;
			kpt.pt = Point2f(points[i][0], points[i][1]);
			kpt.response = responses[i];
			kpt.angle = angles[i];
			kpt.octave = octaves[i];
			kpt.size = sizes[i];
			dst[i] = kpt;
		}
	}

	int descriptorSize() const { return describer_->descriptorSize(); }
	int descriptorType() const { return describer_->descriptorType(); }
	int defaultNorm() const { return describer_->defaultNorm(); }

	void setMaxFeatures(int maxFeatures) { nfeatures_ = maxFeatures; }
	int getMaxFeatures() const { return nfeatures_; }

	void setScaleFactor(float scaleFactor) { scaleFactor_ = scaleFactor; }
	float getScaleFactor() const { return scaleFactor_; }

	void setNLevels(int nlevels) { nlevels_ = nlevels; }
	int getNLevels() const { return nlevels_; }

	void setFirstLevel(int firstLevel) { firstLevel_ = firstLevel; }
	int getFirstLevel() const { return firstLevel_; }

	void setFastThreshold(int fastThreshold) { fastThreshold_ = fastThreshold; }
	int getFastThreshold() const { return fastThreshold_; }

	void setNonmaxRadius(int nonmaxRadius) { nonmaxRadius_ = nonmaxRadius; }
	int getNonmaxRadius() const { return nonmaxRadius_; }

	void setDescriptorType(DescriptorType descriptorType)
	{
		descriptorType_ = descriptorType;
		describer_ = createDescriber(descriptorType);
	}

	DescriptorType getDescriptorType() const { return descriptorType_; }

private:

	int nfeatures_;
	float scaleFactor_;
	int nlevels_;
	int firstLevel_;
	int fastThreshold_;
	int nonmaxRadius_;
	DescriptorType descriptorType_;

	GpuMat image_, keypoints_, descriptors_;
	std::vector<GpuMat> imagePyr_, maskPyr_, kptsPyr_, blurPyr_, descPyr_;

	DeviceBuffer fastBuf_, suppBuf_;
	std::vector<DeviceBuffer> kptsBuf_;

	HostMem h_buffer_;
	int* h_count_;

	std::vector<float> scales_;
	std::vector<int> nfeaturesPerLevel_;
	Ptr<EfficientDescriptorsAsync> describer_;
	Ptr<cuda::Filter> filter_;
};

Ptr<EfficientFeatures> EfficientFeatures::create(int nfeatures, float scaleFactor, int nlevels,
	int firstLevel, int fastThreshold, int nonmaxRadius, DescriptorType dtype)
{
	return makePtr<EfficientFeaturesImpl>(nfeatures, scaleFactor, nlevels,
		firstLevel, fastThreshold, nonmaxRadius, dtype);
}

EfficientFeatures::~EfficientFeatures()
{
}

} // namespace cuda
} // namespace cv
