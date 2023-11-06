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

#ifndef __EFFICIENT_DESCRIPTORS_H__
#define __EFFICIENT_DESCRIPTORS_H__

#include <opencv2/features2d.hpp>

namespace cv
{

/**
 * Implementation of the Box Average Difference (BAD) descriptor. The method uses features
 * computed from the difference of the average gray values of two boxes in the patch.
 *
 * Each pair of boxes is represented with a BoxPairParams struct. After obtaining the feature
 * from them, the i'th feature is thresholded with thresholds_[i], producing the binary
 * weak-descriptor.
 */
class BAD : public Feature2D
{
public:

	/**
	 * @brief  Descriptor number of bits, each bit is a weak-descriptor.
	 * The user can choose between 512 or 256 bits.
	 */
	enum BadSize
	{
		SIZE_512_BITS = 100, SIZE_256_BITS = 101,
	};

	/** @brief Creates the BAD descriptor.
	@param scale_factor Adjust the sampling window around detected keypoints:
	- <b> 1.00f </b> should be the scale for ORB keypoints
	- <b> 6.75f </b> should be the scale for SIFT detected keypoints
	- <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints
	- <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
	@param n_bits Determine the number of bits in the descriptor. Should be either
	  BAD::SIZE_512_BITS or BAD::SIZE_256_BITS.
	*/
	CV_WRAP static Ptr<BAD> create(float scale_factor, int n_bits = BAD::SIZE_512_BITS);
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
class HashSIFT : public Feature2D
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

	/**
	 * Creates a pointer to a new instance
	 * @param cropping_scale Determines the size of the patch cropped for description.
	 * The diameter of the patch will be: cropping_scale * kp.size
	 * @param n_bits The number of bits of the descriptor (512 or 256)
	 * @param sigma The standard deviation of the gaussian smoothing filter applied to the patch
	 * before description.
	 * @return A pointer to the new cv::Feature2D descriptor object
	 */
	CV_WRAP static Ptr<HashSIFT> create(float cropping_scale, int n_bits = SIZE_256_BITS, double sigma = 1.6);
};

} // namespace cv

#endif // !__EFFICIENT_DESCRIPTORS_H__
