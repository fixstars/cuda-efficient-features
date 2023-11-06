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

// Implementation of the article:
//     Iago Suarez, Ghesn Sfeir, Jose M. Buenaposada, and Luis Baumela.
//     Revisiting binary local image description for resource limited devices.
//     IEEE Robotics and Automation Letters, 2021.

#include "efficient_descriptors.h"

#include <opencv2/imgproc.hpp>

namespace cv
{

static const float PI_1_0F = static_cast<float>(CV_PI);
static const float PI_2_0F = static_cast<float>(CV_2PI);

// assumed gaussian blur for input image
static constexpr double SIFT_INIT_SIGMA = 0.5;

// determines the size of a single descriptor orientation histogram
static constexpr float SIFT_DESCR_SCL_FCTR = 3.f;

static constexpr int R_BINS = 4; // width of descriptor histogram array
static constexpr int C_BINS = 4; // width of descriptor histogram array
static constexpr int ORI_BINS = 8; // number of bins per histogram in descriptor array
static constexpr float MAGNITUDE_TH = 0.2f;
static constexpr float INT_DESCR_FACTOR = 512.f;
static constexpr bool STEP1_PYRAMID = false;
static constexpr bool STEP4_GAUSSIAN_WEIGHT = true;
static constexpr bool STEP6_TRILINEAR_INTERP = true;
static constexpr bool STEP7_L2_NORMALIZATION = true;
static constexpr bool STEP8_TRIM_BIGVALS = true;
static constexpr bool STEP9_UCHAR_SCALING = true;
static constexpr bool USE_BICUBIC_INTERPOLATION = false;

static void convertToGray(const Mat& src, Mat& dst)
{
	switch (src.type()) {
	case CV_8UC1:
		dst = src;
		break;
	case CV_8UC3:
		cvtColor(src, dst, COLOR_BGR2GRAY);
		break;
	case CV_8UC4:
		cvtColor(src, dst, COLOR_BGRA2GRAY);
		break;
	default:
		CV_Error(Error::StsBadArg, "Image should be 8UC1, 8UC3 or 8UC4");
	}
}

static void warpAffineLinear(const Mat& src, Mat& dst, const Matx23f& M, Size dstSize)
{
	CV_Assert(src.type() == CV_8U);

	dst.create(dstSize, CV_8U);

	const float M00 = M(0, 0);
	const float M01 = M(0, 1);
	const float M02 = M(0, 2);
	const float M10 = M(1, 0);
	const float M11 = M(1, 1);
	const float M12 = M(1, 2);

	const size_t step = src.step;

	for (int y = 0; y < dst.rows; y++)
	{
		uchar* ptrDst = dst.ptr<uchar>(y);
		for (int x = 0; x < dst.cols; x++)
		{
			const float u = M00 * x + M01 * y + M02;
			const float v = M10 * x + M11 * y + M12;

			uchar dstVal = 0;
			const int ui = cvFloor(u);
			const int vi = cvFloor(v);
			if (ui >= 0 && ui + 1 < src.cols && vi >= 0 && vi + 1 < src.rows)
			{
				const uchar* ptrSrc = src.data + vi * step + ui;

				const float du = u - ui;
				const float dv = v - vi;
				const float tmp0 = (1 - du) * ptrSrc[0] + du * ptrSrc[1];
				const float tmp1 = (1 - du) * ptrSrc[step] + du * ptrSrc[step + 1];
				const float tmp2 = (1 - dv) * tmp0 + dv * tmp1;
				dstVal = static_cast<uchar>(std::min(static_cast<int>(tmp2 + 0.5f), 255));
			}

			ptrDst[x] = dstVal;
		}
	}
}

static void rectifyPatch(const Mat& image, const KeyPoint& kp, Mat& patch, Size patchSize, float scaleFactor)
{
	const int w = patchSize.width;
	const int h = patchSize.height;

	patch.create(h, w, CV_8U);

	const float s = scaleFactor * kp.size / (0.5f * (w + h));
	const float theta = PI_1_0F * kp.angle / 180;

	const float cost = s * (kp.angle >= 0 ? cosf(theta) : 1.f);
	const float sint = s * (kp.angle >= 0 ? sinf(theta) : 0.f);

	Matx23f M;

	M(0, 0) = +cost;
	M(0, 1) = -sint;
	M(0, 2) = (-cost + sint) * patchSize.width / 2.f + kp.pt.x;

	M(1, 0) = +sint;
	M(1, 1) = +cost;
	M(1, 2) = (-sint - cost) * patchSize.height / 2.f + kp.pt.y;

	if (USE_BICUBIC_INTERPOLATION)
		warpAffine(image, patch, M, patchSize, WARP_INVERSE_MAP + INTER_CUBIC, BORDER_REPLICATE);
	else
		warpAffineLinear(image, patch, M, patchSize);
}

static inline float squared(float x)
{
	return x * x;
}

static inline float normsq(float x, float y)
{
	return squared(x) + squared(y);
}

static void normalize(float* desc, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += squared(desc[i]);

	const float norm = std::max(sqrtf(sum), FLT_EPSILON);
	const float scale = 1.f / norm;
	for (int i = 0; i < size; i++)
		desc[i] *= scale;
}

struct HistBin
{
	HistBin(int h, int w, float kpScale)
	{
		const float cellh = SIFT_DESCR_SCL_FCTR * (kpScale * h * 0.5f);
		const float cellw = SIFT_DESCR_SCL_FCTR * (kpScale * w * 0.5f);

		scaleR = 1.f / cellh;
		scaleC = 1.f / cellw;
		scaleO = ORI_BINS / PI_2_0F;

		halfh = 0.5f * h;
		halfw = 0.5f * w;
		rbin0 = R_BINS / 2 - 0.5f;
		cbin0 = C_BINS / 2 - 0.5f;
	}

	inline float getRBin(int r) const { return scaleR * (r - halfh) + rbin0; }
	inline float getCBin(int c) const { return scaleC * (c - halfw) + cbin0; }
	inline float getOBin(int o) const { return scaleO * o; }

	float scaleR, scaleC, scaleO, halfh, halfw, rbin0, cbin0;
};

static inline std::pair<int, float> separateIF(float value)
{
	const int vi = cvFloor(value);
	const float vf = value - vi;
	return { vi, vf };
}

static inline std::pair<float, float> distribute(float value, float weight)
{
	const float v1 = weight * value;
	const float v0 = value - v1;
	return { v0, v1 };
}

static void computePatchSIFT(const Mat& patch, float* descriptors, int descriptorSize,
	float kpScale, double sigma)
{
	// Step 1: Filter the image using a gaussian filter of size sigma
	Mat img;
	if (STEP1_PYRAMID)
		GaussianBlur(patch, img, Size(), sigma, sigma);
	else
		img = patch;

	// Step 2: We assume a pre-oriented and pre-scale patch of fixed size
	// Step 3: Compute the image derivatives
	// Step 4: Create the gaussian weighting assigned to each pixel
	// Step 5: Build the histogram of orientations
	// Step6: Histogram update using tri-linear interpolation
	const int h = img.rows;
	const int w = img.cols;
	const int dh = h - 2;
	const int dw = w - 2;

	const float kpRadius = kpScale * h * 0.5f;
	const float kernelSigma = 0.5f * C_BINS * SIFT_DESCR_SCL_FCTR * kpRadius;
	const float distScale = -1.f / (2 * kernelSigma * kernelSigma);
	const float cx = 0.5f * dw;
	const float cy = 0.5f * dh;

	const int histShape[3] = { R_BINS + 2, C_BINS + 2, ORI_BINS + 2 };
	Mat hist(3, histShape, CV_32F);
	const size_t histStep1 = hist.step[1] / sizeof(float);
	hist = 0;

	HistBin histBin(h, w, kpScale);

	for (int y = 0; y < dh; y++)
	{
		const uchar* pT = img.ptr<uchar>(y + 0) + 1;
		const uchar* pC = img.ptr<uchar>(y + 1) + 1;
		const uchar* pB = img.ptr<uchar>(y + 2) + 1;

		const auto[ri, rf] = separateIF(histBin.getRBin(y + 1));

		float* ptrHist0 = hist.ptr<float>(ri + 1);
		float* ptrHist1 = hist.ptr<float>(ri + 2);

		for (int x = 0; x < dw; x++)
		{
			// Multiply the gradient magnitude by the importance of each pixel
			const float magScale = expf(distScale * normsq(x - cx, y - cy));

			// Compute the derivative using the previous and next pixels
			const float dx = pC[x + 1] - pC[x - 1];
			const float dy = pT[x] - pB[x];

			const float mag = magScale * sqrtf(normsq(dx, dy));
			const float ori = atan2f(dy, dx);

			const auto[ci, cf] = separateIF(histBin.getCBin(x + 1));

			auto[oi, of] = separateIF(histBin.scaleO * ori);
			if (oi < 0) oi += ORI_BINS;
			if (oi >= ORI_BINS) oi -= ORI_BINS;

			// distribute along r
			const auto[v0, v1] = distribute(mag, rf);

			// distribute along c
			const auto[v00, v01] = distribute(v0, cf);
			const auto[v10, v11] = distribute(v1, cf);

			// distribute along o
			const auto[v000, v001] = distribute(v00, of);
			const auto[v010, v011] = distribute(v01, of);
			const auto[v100, v101] = distribute(v10, of);
			const auto[v110, v111] = distribute(v11, of);

			float* ptrHist00 = ptrHist0 + (ci + 1) * histStep1;
			float* ptrHist01 = ptrHist0 + (ci + 2) * histStep1;
			float* ptrHist10 = ptrHist1 + (ci + 1) * histStep1;
			float* ptrHist11 = ptrHist1 + (ci + 2) * histStep1;

			ptrHist00[oi + 0] += v000;
			ptrHist00[oi + 1] += v001;
			ptrHist01[oi + 0] += v010;
			ptrHist01[oi + 1] += v011;

			ptrHist10[oi + 0] += v100;
			ptrHist10[oi + 1] += v101;
			ptrHist11[oi + 0] += v110;
			ptrHist11[oi + 1] += v111;
		}
	}

	// Finalize histogram, since the orientation histograms are circular
	for (int r = 0; r < R_BINS; r++)
	{
		float* ptrHist = hist.ptr<float>(r + 1) + histStep1;
		for (int c = 0; c < C_BINS; c++, ptrHist += histStep1)
		{
			// Increase the value in the penultimate orientation bin in the first one
			ptrHist[0] += ptrHist[ORI_BINS + 0];

			// Increase the value in last orientation bin in the second one
			ptrHist[1] += ptrHist[ORI_BINS + 1];

			// Copy the values in the histogram to the output destination
			for (int k = 0; k < ORI_BINS; k++)
				descriptors[(r * R_BINS + c) * ORI_BINS + k] = ptrHist[k];
		}
	}

	// Step 7: Apply L2 normalization
	if (STEP7_L2_NORMALIZATION)
	{
		normalize(descriptors, descriptorSize);
	}

	// Step 8: Trim Big Values
	if (STEP8_TRIM_BIGVALS)
	{
		for (int i = 0; i < descriptorSize; i++)
			descriptors[i] = std::min(descriptors[i], MAGNITUDE_TH);

		normalize(descriptors, descriptorSize);
	}

	// Optional Step 9: Scale the result, so that it can be easily converted to byte array
	if (STEP9_UCHAR_SCALING)
	{
		for (int k = 0; k < descriptorSize; k++)
			descriptors[k] = saturate_cast<uchar>(INT_DESCR_FACTOR * descriptors[k]);
	}
}

static void computePatchSIFTs(const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& responses,
	Size patchSize, float croppingScale, float keypointScale = 1.f / 6, double sigma = 1.6)
{
	const int npoints = static_cast<int>(keypoints.size());

	sigma = sqrt(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01));

	responses.create(npoints, 129, CV_32F);

	Mat patch;
	for (int i = 0; i < npoints; i++)
	{
		float* response = responses.ptr<float>(i);
		response[0] = 1;

		rectifyPatch(image, keypoints[i], patch, patchSize, croppingScale);
		computePatchSIFT(patch, response + 1, 128, keypointScale, sigma);
	}
}

static void matmulAndSign(const Mat& responses, const Mat& b_matrix, Mat& descriptors)
{
	CV_Assert(responses.rows == descriptors.rows);

	Mat tmp;
	gemm(responses, b_matrix, 1, Mat(), 0, tmp, GEMM_2_T);

	for (int i = 0; i < descriptors.rows; i++)
	{
		const float* ptrTmp = tmp.ptr<float>(i);
		uchar* ptrDesc = descriptors.ptr<uchar>(i);
		for (int b = 0; b < descriptors.cols; b++)
		{
			uchar byte = 0;
			byte |= (*ptrTmp++ > 0) << 7;
			byte |= (*ptrTmp++ > 0) << 6;
			byte |= (*ptrTmp++ > 0) << 5;
			byte |= (*ptrTmp++ > 0) << 4;
			byte |= (*ptrTmp++ > 0) << 3;
			byte |= (*ptrTmp++ > 0) << 2;
			byte |= (*ptrTmp++ > 0) << 1;
			byte |= (*ptrTmp++ > 0) << 0;
			ptrDesc[b] = byte;
		}
	}
}

class HashSIFTImpl : public HashSIFT
{
public:

	HashSIFTImpl(float croppingScale, int n_bits, double sigma) : croppingScale_(croppingScale)
	{
#include "hash_sift.p512.h"
#include "hash_sift.p256.h"

		if (n_bits == SIZE_512_BITS)
			Mat(512, 129, CV_64F, (void*)HASH_SIFT_512_VALS).convertTo(b_matrix_, CV_32F);
		else if (n_bits == SIZE_256_BITS)
			Mat(256, 129, CV_64F, (void*)HASH_SIFT_256_VALS).convertTo(b_matrix_, CV_32F);
		else
			CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");

		n_bits_ = b_matrix_.rows;
	}

	void compute(InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors) CV_OVERRIDE
	{
		Mat image = _image.getMat();
		if (image.empty())
			return;

		if (keypoints.empty())
		{
			// clean output buffer (it may be reused with "allocated" data)
			_descriptors.release();
			return;
		}

		// create the output array of descriptors
		_descriptors.create((int)keypoints.size(), descriptorSize(), descriptorType());
		Mat descriptors = _descriptors.getMat();

		// convert to gray 
		Mat grayImage;
		convertToGray(image, grayImage);

		// compute SIFT
		Mat responses;
		computePatchSIFTs(grayImage, keypoints, responses, Size(32, 32), croppingScale_);

		// compute the linear projection and hash
		matmulAndSign(responses, b_matrix_, descriptors);
	}

	int descriptorSize() const override { return n_bits_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float croppingScale_;
	Mat b_matrix_;
	int n_bits_;
};

Ptr<HashSIFT> HashSIFT::create(float croppingScale, int n_bits, double sigma)
{
	return makePtr<HashSIFTImpl>(croppingScale, n_bits, sigma);
}

} // namespace cv
