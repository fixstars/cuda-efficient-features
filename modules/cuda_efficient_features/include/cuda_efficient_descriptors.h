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

#ifndef __CUDA_EFFICIENT_DESCRIPTORS_H__
#define __CUDA_EFFICIENT_DESCRIPTORS_H__

#include <opencv2/core/cuda.hpp>

namespace cv
{
namespace cuda
{

class EfficientDescriptorsAsync
{
public:

	/** @brief Computes the descriptors for a set of keypoints detected in an image.

	@param image Image.
	@param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
	computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
	with several dominant orientations (for each orientation).
	@param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
	descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
	descriptor for keypoint j-th keypoint.
	 */
	virtual void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) = 0;

	/** @brief Computes the descriptors for a set of keypoints detected in an image.

	@param image Image.
	@param keypoints Input collection of keypoints.
	@param descriptors Computed descriptors. Row j is the descriptor for j-th keypoint.
	@param stream CUDA stream.
	 */
	virtual void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream = Stream::Null()) = 0;

	virtual int descriptorSize() const = 0;
	virtual int descriptorType() const = 0;
	virtual int defaultNorm() const = 0;

	virtual ~EfficientDescriptorsAsync();
};

/**
 * Implementation of the Box Average Difference (BAD) descriptor. The method uses features
 * computed from the difference of the average gray values of two boxes in the patch.
 *
 * Each pair of boxes is represented with a BoxPairParams struct. After obtaining the feature
 * from them, the i'th feature is thresholded with thresholds_[i], producing the binary
 * weak-descriptor.
 */
class BAD : public EfficientDescriptorsAsync
{
public:

	/**
	 * @brief  Descriptor number of bits, each bit is a weak-descriptor.
	 * The user can choose between 512 or 256 bits.
	 */
	enum BADSize
	{
		SIZE_512_BITS = 100, SIZE_256_BITS = 101,
	};

	/** @brief Creates the BAD descriptor.
	@param scaleFactor Adjust the sampling window around detected keypoints:
	- <b> 1.00f </b> should be the scale for ORB keypoints
	- <b> 6.75f </b> should be the scale for SIFT detected keypoints
	- <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints
	- <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
	@param nbits Determine the number of bits in the descriptor. Should be either
	BAD::SIZE_512_BITS or BAD::SIZE_256_BITS.
	 */
	static Ptr<BAD> create(float scaleFactor, int nbits = SIZE_256_BITS);
};

/**
 * HashSIFT descriptor described in the article
 * "Suarez, I., Buenaposada, J. M., & Baumela, L. (2021). Revisiting Binary Local Image
 * Description for Resource Limited Devices. IEEE Robotics and Automation Letters."
 *
 * The descriptor computes first the SIFT features using the class PatchSIFT and then it hashes
 * the floating point SIFT descriptor with a pre-learnt linear projection: b_matrix_
 * The weights b_matrix_ are loaded from the files HashSiftWeights256.i and HashSiftWeights512.i.
 */
class HashSIFT : public EfficientDescriptorsAsync
{
public:

	/**
	 * @brief  Descriptor number of bits, each bit is a weak-descriptor.
	 * The user can choose between 512 or 256 bits.
	 */
	enum HashSIFTSize
	{
		SIZE_512_BITS = 100, SIZE_256_BITS = 101,
	};

	/** @brief Creates the HashSIFT descriptor.
	@param croppingScale Determines the size of the patch cropped for description.
	The diameter of the patch will be: croppingScale * kp.size
	@param nbits Determine the number of bits in the descriptor. Should be either
	HashSIFT::SIZE_512_BITS or HashSIFT::SIZE_256_BITS.
	 */
	static Ptr<HashSIFT> create(float croppingScale, int nbits = SIZE_256_BITS);
};

} // namespace cuda
} // namespace cv

#endif // !__CUDA_EFFICIENT_DESCRIPTORS_H__
