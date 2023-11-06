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

#define CV_BAD_PARALLEL

#define CV_ROUNDNUM(x) ((int)(x + 0.5f))
#define CV_DEGREES_TO_RADS 0.017453292519943295 // (M_PI / 180.0)
#define CV_BAD_EXTRA_RATIO_MARGIN 1.75f

using namespace cv;
using namespace std;

namespace cv
{

// Struct containing the 6 parameters that define an Average Box weak-learner
struct BoxPairParams
{
	int x1, x2, y1, y2, boxRadius;
};

// BAD implementation
class BAD_Impl CV_FINAL : public BAD
{
public:

	// constructor
	explicit BAD_Impl(float scale_factor, int n_bits = SIZE_512_BITS);

	// destructor
	~BAD_Impl() CV_OVERRIDE = default;

	// returns the descriptor length in bytes
	int descriptorSize() const CV_OVERRIDE { return int(box_pair_params_.size() / 8); }

	// returns the descriptor type
	int descriptorType() const CV_OVERRIDE { return CV_8UC1; }

	// returns the default norm type
	int defaultNorm() const CV_OVERRIDE { return cv::NORM_HAMMING; }

	// compute descriptors given keypoints
	void compute(InputArray image, vector<KeyPoint> &keypoints, OutputArray descriptors) CV_OVERRIDE;

private:
	std::vector<BoxPairParams> box_pair_params_;
	std::vector<float> thresholds_;
	float scale_factor_;
	cv::Size patch_size_;

	void computeBAD(const cv::Mat &integralImg,
		const std::vector<cv::KeyPoint> &keypoints,
		cv::Mat &descriptors);
}; // END BAD_Impl CLASS

/**
 * @brief Function that determines if a keypoint is close to the image border.
 * @param kp The detected keypoint
 * @param imgSize The size of the image
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @return true if the keypoint is in the border, false otherwise
 */
static inline bool isKeypointInTheBorder(const cv::KeyPoint &kp,
	const cv::Size &imgSize,
	const cv::Size &patchSize = { 32, 32 },
	float scaleFactor = 1)
{
	// This would be the correct measure but since we will compare with half of the size, use this as border size
	float s = scaleFactor * kp.size / (patchSize.width + patchSize.height);
	cv::Size2f border(patchSize.width * s * CV_BAD_EXTRA_RATIO_MARGIN,
		patchSize.height * s * CV_BAD_EXTRA_RATIO_MARGIN);

	if (kp.pt.x < border.width || kp.pt.x + border.width >= imgSize.width)
		return true;

	if (kp.pt.y < border.height || kp.pt.y + border.height >= imgSize.height)
		return true;

	return false;
}

/**
 * @brief Rectifies the coordinates of the box pairs measurement functions with the keypoint
 * location parameters.
 * @param boxes_params The input parameters defining the location of each pair of boxes inside
 * the normalized (32x32) patch.
 * @param out_params The output parameters, now the boxes are located in the full image.
 * @param kp The keypoint defining the offset, rotation and scale to be applied
 * @param scale_factor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @param patch_size The size of the normalized patch where the measurement functions were learnt.
 */
static inline void rectifyBoxes(const std::vector<BoxPairParams> &boxesParams,
	std::vector<BoxPairParams> &outParams,
	const cv::KeyPoint &kp,
	float scaleFactor = 1,
	const cv::Size &patchSize = cv::Size(32, 32))
{
	float m00, m01, m02, m10, m11, m12;
	float s, cosine, sine;

	s = scaleFactor * kp.size / (0.5f * (patchSize.width + patchSize.height));
	outParams.resize(boxesParams.size());

	if (kp.angle == -1)
	{
		m00 = s;
		m01 = 0.0f;
		m02 = -0.5f * s * patchSize.width + kp.pt.x;
		m10 = 0.0f;
		m11 = s;
		m12 = -s * 0.5f * patchSize.height + kp.pt.y;
	}
	else
	{
		cosine = (kp.angle >= 0) ? float(cos(kp.angle * CV_DEGREES_TO_RADS)) : 1.f;
		sine = (kp.angle >= 0) ? float(sin(kp.angle * CV_DEGREES_TO_RADS)) : 0.f;

		m00 = s * cosine;
		m01 = -s * sine;
		m02 = (-s * cosine + s * sine) * patchSize.width * 0.5f + kp.pt.x;
		m10 = s * sine;
		m11 = s * cosine;
		m12 = (-s * sine - s * cosine) * patchSize.height * 0.5f + kp.pt.y;
	}

	for (size_t i = 0; i < boxesParams.size(); i++)
	{
		outParams[i].x1 = CV_ROUNDNUM(m00 * boxesParams[i].x1 + m01 * boxesParams[i].y1 + m02);
		outParams[i].y1 = CV_ROUNDNUM(m10 * boxesParams[i].x1 + m11 * boxesParams[i].y1 + m12);
		outParams[i].x2 = CV_ROUNDNUM(m00 * boxesParams[i].x2 + m01 * boxesParams[i].y2 + m02);
		outParams[i].y2 = CV_ROUNDNUM(m10 * boxesParams[i].x2 + m11 * boxesParams[i].y2 + m12);
		outParams[i].boxRadius = CV_ROUNDNUM(s * boxesParams[i].boxRadius);
	}
}

/**
 * @brief Computes the Box Average Difference, measuring the difference of gray level in the two
 * square regions.
 * @param box_params The weak-learner parameter defining the size and locations of each box.
 * @param integral_img The integral image used to compute the average gray value in the square regions.
 * @return The difference of gray level in the two squares defined by box_params
 */
static inline float computeBadResponse(const BoxPairParams &boxParams,
	const cv::Mat &integralImage)
{
	CV_DbgAssert(!integralImage.empty());
	CV_DbgAssert(integralImage.type() == CV_32SC1);

	int frameWidth, frameHeight, box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2;
	int A, B, C, D;
	int box_area1, box_area2;
	float sum1, sum2, average1, average2;
	// Since the integral image has one extra row and col, calculate the patch dimensions
	frameWidth = integralImage.cols;
	frameHeight = integralImage.rows;

	// For the first box, we calculate its margin coordinates
	box1x1 = boxParams.x1 - boxParams.boxRadius;
	if (box1x1 < 0)
		box1x1 = 0;
	else if (box1x1 >= frameWidth - 1)
		box1x1 = frameWidth - 2;
	box1y1 = boxParams.y1 - boxParams.boxRadius;
	if (box1y1 < 0)
		box1y1 = 0;
	else if (box1y1 >= frameHeight - 1)
		box1y1 = frameHeight - 2;
	box1x2 = boxParams.x1 + boxParams.boxRadius + 1;
	if (box1x2 <= 0)
		box1x2 = 1;
	else if (box1x2 >= frameWidth)
		box1x2 = frameWidth - 1;
	box1y2 = boxParams.y1 + boxParams.boxRadius + 1;
	if (box1y2 <= 0)
		box1y2 = 1;
	else if (box1y2 >= frameHeight)
		box1y2 = frameHeight - 1;
	CV_DbgAssert((box1x1 < box1x2 && box1y1 < box1y2) && "Box 1 has size 0");

	// For the second box, we calculate its margin coordinates
	box2x1 = boxParams.x2 - boxParams.boxRadius;
	if (box2x1 < 0)
		box2x1 = 0;
	else if (box2x1 >= frameWidth - 1)
		box2x1 = frameWidth - 2;
	box2y1 = boxParams.y2 - boxParams.boxRadius;
	if (box2y1 < 0)
		box2y1 = 0;
	else if (box2y1 >= frameHeight - 1)
		box2y1 = frameHeight - 2;
	box2x2 = boxParams.x2 + boxParams.boxRadius + 1;
	if (box2x2 <= 0)
		box2x2 = 1;
	else if (box2x2 >= frameWidth)
		box2x2 = frameWidth - 1;
	box2y2 = boxParams.y2 + boxParams.boxRadius + 1;
	if (box2y2 <= 0)
		box2y2 = 1;
	else if (box2y2 >= frameHeight)
		box2y2 = frameHeight - 1;
	CV_DbgAssert((box2x1 < box2x2 && box2y1 < box2y2) && "Box 2 has size 0");

	// Read the integral image values for the first box
	A = integralImage.at<int>(box1y1, box1x1);
	B = integralImage.at<int>(box1y1, box1x2);
	C = integralImage.at<int>(box1y2, box1x1);
	D = integralImage.at<int>(box1y2, box1x2);

	// Calculate the mean intensity value of the pixels in the box
	sum1 = float(A + D - B - C);
	box_area1 = (box1y2 - box1y1) * (box1x2 - box1x1);
	CV_DbgAssert(box_area1 > 0);
	average1 = sum1 / box_area1;

	// Calculate the indices on the integral image where the box falls
	A = integralImage.at<int>(box2y1, box2x1);
	B = integralImage.at<int>(box2y1, box2x2);
	C = integralImage.at<int>(box2y2, box2x1);
	D = integralImage.at<int>(box2y2, box2x2);

	// Calculate the mean intensity value of the pixels in the box
	sum2 = float(A + D - B - C);
	box_area2 = (box2y2 - box2y1) * (box2x2 - box2x1);
	CV_DbgAssert(box_area2 > 0);
	average2 = sum2 / box_area2;

	return average1 - average2;
}

// descriptor computation using keypoints
void BAD_Impl::compute(InputArray _image, vector<KeyPoint> &keypoints, OutputArray _descriptors)
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

	Mat grayImage;
	switch (image.type()) {
	case CV_8UC1:
		grayImage = image;
		break;
	case CV_8UC3:
		cvtColor(image, grayImage, COLOR_BGR2GRAY);
		break;
	case CV_8UC4:
		cvtColor(image, grayImage, COLOR_BGRA2GRAY);
		break;
	default:
		CV_Error(Error::StsBadArg, "Image should be 8UC1, 8UC3 or 8UC4");
	}

	cv::Mat integralImg;

	// compute the integral image
	cv::integral(grayImage, integralImg);

	// Create the output array of descriptors
	_descriptors.create((int)keypoints.size(), descriptorSize(), descriptorType());

	// descriptor storage
	cv::Mat descriptors = _descriptors.getMat();
	CV_DbgAssert(descriptors.type() == CV_8UC1);

	// Compute the BAD descriptors
	computeBAD(integralImg, keypoints, descriptors);
}

// constructor
BAD_Impl::BAD_Impl(float scale_factor, int n_bits)
	: scale_factor_(scale_factor), patch_size_(32, 32)
{
#include "bad.p512.h"
#include "bad.p256.h"
	if (n_bits == SIZE_512_BITS)
	{
		box_pair_params_.assign(box_pair_params_512, box_pair_params_512 + sizeof(box_pair_params_512) / sizeof(box_pair_params_512[0]));
		thresholds_.assign(thresholds_512, thresholds_512 + 512);
	}
	else if (n_bits == SIZE_256_BITS)
	{
		box_pair_params_.assign(box_pair_params_256, box_pair_params_256 + sizeof(box_pair_params_256) / sizeof(box_pair_params_256[0]));
		thresholds_.assign(thresholds_256, thresholds_256 + 256);
	}
	else
		CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");
}

// Internal function that implements the core of BAD descriptor
void BAD_Impl::computeBAD(const cv::Mat &integralImg,
	const std::vector<cv::KeyPoint> &keypoints,
	cv::Mat &descriptors)
{
	CV_DbgAssert(!integralImg.empty());
	CV_DbgAssert(size_t(descriptors.rows) == keypoints.size());
	const int *integralPtr = integralImg.ptr<int>();
	cv::Size frameSize(integralImg.cols - 1, integralImg.rows - 1);

	const int kpSize = static_cast<int>(keypoints.size());

	// Get a pointer to the first element in the range
	BoxPairParams *boxPair;
	float responseFun;
	int areaResponseFun;
	size_t boxIdx;
	int box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2, bit_idx, side;
	uchar byte = 0;
	std::vector<BoxPairParams> imgWLParams(box_pair_params_.size());
	uchar *d = descriptors.ptr<uchar>();

	for (int kpIdx = 0; kpIdx < kpSize; kpIdx++)
	{
		// Rectify the weak learners coordinates using the keypoint information
		rectifyBoxes(box_pair_params_, imgWLParams, keypoints[kpIdx], scale_factor_, patch_size_);
		if (isKeypointInTheBorder(keypoints[kpIdx], frameSize, patch_size_, scale_factor_))
		{
			// Code to process the keypoints in the image margins
			for (boxIdx = 0; boxIdx < box_pair_params_.size(); boxIdx++) {
				bit_idx = 7 - int(boxIdx % 8);
				responseFun = computeBadResponse(imgWLParams[boxIdx], integralImg);
				// Set the bit to 1 if the response function is less or equal to the threshod
				byte |= (responseFun <= thresholds_[boxIdx]) << bit_idx;
				// If we filled the byte, save it
				if (bit_idx == 0)
				{
					*d = byte;
					byte = 0;
					d++;
				}
			}
		}
		else
		{
			// Code to process the keypoints in the image center
			boxPair = imgWLParams.data();
			for (boxIdx = 0; boxIdx < box_pair_params_.size(); boxIdx++)
			{
				bit_idx = 7 - int(boxIdx % 8);

				// For the first box, we calculate its margin coordinates
				box1x1 = boxPair->x1 - boxPair->boxRadius;
				box1y1 = (boxPair->y1 - boxPair->boxRadius) * integralImg.cols;
				box1x2 = boxPair->x1 + boxPair->boxRadius + 1;
				box1y2 = (boxPair->y1 + boxPair->boxRadius + 1) * integralImg.cols;
				// For the second box, we calculate its margin coordinates
				box2x1 = boxPair->x2 - boxPair->boxRadius;
				box2y1 = (boxPair->y2 - boxPair->boxRadius) * integralImg.cols;
				box2x2 = boxPair->x2 + boxPair->boxRadius + 1;
				box2y2 = (boxPair->y2 + boxPair->boxRadius + 1) * integralImg.cols;
				side = 1 + (boxPair->boxRadius << 1);

				// Get the difference between the average level of the two boxes
				areaResponseFun = (integralPtr[box1y1 + box1x1]  // A of Box1
					+ integralPtr[box1y2 + box1x2]               // D of Box1
					- integralPtr[box1y1 + box1x2]               // B of Box1
					- integralPtr[box1y2 + box1x1]               // C of Box1
					- integralPtr[box2y1 + box2x1]               // A of Box2
					- integralPtr[box2y2 + box2x2]               // D of Box2
					+ integralPtr[box2y1 + box2x2]               // B of Box2
					+ integralPtr[box2y2 + box2x1]);             // C of Box2

				// Set the bit to 1 if the response function is less or equal to the threshod
				byte |= (areaResponseFun <= (thresholds_[boxIdx] * (side * side))) << bit_idx;
				boxPair++;
				// If we filled the byte, save it
				if (bit_idx == 0)
				{
					*d = byte;
					byte = 0;
					d++;
				}
			}  // End of for each dimension
		}  // End of else (of pixels in the image center)
	}  // End of for each keypoint
}

Ptr<BAD> BAD::create(float scale_factor, int n_bits)
{
	return makePtr<BAD_Impl>(scale_factor, n_bits);
}

} // END NAMESPACE CV
