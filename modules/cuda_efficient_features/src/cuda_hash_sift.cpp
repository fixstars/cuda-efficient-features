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

#include "cuda_efficient_descriptors.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cublas_v2.h>

#include "cuda_hash_sift_internal.h"
#include "cuda_efficient_features_internal.h"
#include "device_buffer.h"

namespace cv
{
namespace cuda
{

#define CUBLAS_CHECK(err) \
do {\
	if (err != CUBLAS_STATUS_SUCCESS) { \
		printf("[CUBLAS Error] (code: %d) at %s:%d\n", err, __FILE__, __LINE__); \
	} \
} while (0)

static void hashSIFTGemm(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const cublasHandle_t& handle)
{
	CV_Assert( src1.type() == CV_32FC1 );
	CV_Assert( src1.cols == src2.cols );

	const float alphaf = 1.0f;
	const float betaf = 0.0f;
	const cublasOperation_t transa = CUBLAS_OP_T;
	const cublasOperation_t transb = CUBLAS_OP_N;

	CUBLAS_CHECK( cublasSgemm_v2(handle, transa, transb, src2.rows, src1.rows, src2.cols,
		&alphaf,
		src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
		src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
		&betaf,
		dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
}

class MatmulAndSign
{
public:

	MatmulAndSign()
	{
		CUBLAS_CHECK( cublasCreate_v2(&handle_) );
		CUBLAS_CHECK( cublasSetPointerMode_v2(handle_, CUBLAS_POINTER_MODE_HOST) );
	}

	~MatmulAndSign()
	{
		CUBLAS_CHECK( cublasDestroy_v2(handle_) );
	}

	void operator()(const GpuMat& responses, const GpuMat& bMatrix, GpuMat& descriptors, Stream& stream)
	{
		CV_Assert(responses.rows == descriptors.rows);
		CUBLAS_CHECK( cublasSetStream_v2(handle_, StreamAccessor::getStream(stream)) );

		GpuMat tmp = bufTmp_.createMat(responses.rows, bMatrix.rows, responses.type());
		hashSIFTGemm(responses, bMatrix, tmp, handle_);
		gpu::binarizeDescriptors(tmp, descriptors, StreamAccessor::getStream(stream));
	}

private:

	DeviceBuffer bufTmp_;
	cublasHandle_t handle_;
};

class HashSIFTImpl : public HashSIFT
{
public:

	HashSIFTImpl(float croppingScale, int nbits) : croppingScale_(croppingScale)
	{
#include "hash_sift.p512.h"
#include "hash_sift.p256.h"

		if (nbits == SIZE_512_BITS)
			Mat(512, 129, CV_64F, (void*)HASH_SIFT_512_VALS).convertTo(bMatrix_, CV_32F);
		else if (nbits == SIZE_256_BITS)
			Mat(256, 129, CV_64F, (void*)HASH_SIFT_256_VALS).convertTo(bMatrix_, CV_32F);
		else
			CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");

		nbits_ = bMatrix_.rows;
		d_bMatrix_.upload(bMatrix_);
	}

	void computeHashSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
	{
		if (_image.empty())
			return;

		if (isEmpty(_keypoints))
		{
			// clean output buffer (it may be reused with "allocated" data)
			_descriptors.release();
			return;
		}

		CV_Assert(_image.type() == CV_8U);

		getInputMat(_image, image_, stream);
		getKeypointsMat(_keypoints, keypoints_, stream);
		getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

		GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
		gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, StreamAccessor::getStream(stream));
		matmulAndSign_(responses, d_bMatrix_, descriptors_, stream);

		if (_descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors);
	}

	void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, Stream::Null());
	}

	void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, stream);
	}

	int descriptorSize() const override { return nbits_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float croppingScale_;
	Mat bMatrix_;
	int nbits_;

	GpuMat image_, keypoints_, descriptors_, d_bMatrix_;
	DeviceBuffer bufResponses_;
	MatmulAndSign matmulAndSign_;
};

Ptr<HashSIFT> HashSIFT::create(float croppingScale, int nbits)
{
	return makePtr<HashSIFTImpl>(croppingScale, nbits);
}

} // namespace cuda
} // namespace cv
