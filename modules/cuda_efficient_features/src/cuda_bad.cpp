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

#include "cuda_bad_internal.h"
#include "cuda_efficient_features_internal.h"
#include "device_buffer.h"

namespace cv
{
namespace cuda
{

class BADImpl : public BAD
{
public:

	BADImpl(float scaleFactor, int nbits) : scaleFactor_(scaleFactor), nbits_(nbits), patchSize_(32, 32)
	{
		paramSize_ = nbits == SIZE_256_BITS ? 256 : 512;
		gpu::loadBoxPairParams(paramSize_);
	}

	void computeBAD(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
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

		integral_ = buf_.createMat(image_.rows + 1, image_.cols + 1, CV_32S);
		gpu::calcIntegralImage(image_, integral_, stream);
		gpu::computeBAD(integral_, keypoints_, descriptors_, scaleFactor_, paramSize_, patchSize_, StreamAccessor::getStream(stream));

		if (_descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors);
	}

	void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
	{
		computeBAD(_image, _keypoints, _descriptors, Stream::Null());
	}

	void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
	{
		computeBAD(_image, _keypoints, _descriptors, stream);
	}

	int descriptorSize() const override { return paramSize_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float scaleFactor_;
	int nbits_;
	Size patchSize_;
	int paramSize_;

	GpuMat image_, keypoints_, descriptors_, integral_;
	DeviceBuffer buf_;
};

Ptr<BAD> BAD::create(float scaleFactor, int nbits)
{
	return makePtr<BADImpl>(scaleFactor, nbits);
}

} // namespace cuda
} // namespace cv
