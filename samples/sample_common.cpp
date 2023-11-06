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

#include "sample_common.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::cuda::EfficientFeatures::DescriptorType getDescriptorType(int descType, int descBits)
{
	using namespace cv::cuda;

	if (descType == BAD)
		return descBits == 256 ? EfficientFeatures::BAD_256 : EfficientFeatures::BAD_512;

	if (descType == HashSIFT)
		return descBits == 256 ? EfficientFeatures::HASH_SIFT_256 : EfficientFeatures::HASH_SIFT_512;

	return EfficientFeatures::HASH_SIFT_256;
}

void convertToGray(const cv::Mat& src, cv::Mat& dst)
{
	if (src.type() == CV_8U)
		dst = src;
	else if (src.type() == CV_8UC3)
		cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	else if (src.type() == CV_8UC4)
		cv::cvtColor(src, dst, cv::COLOR_BGRA2GRAY);
	else
		CV_Error(cv::Error::StsBadArg, "Image should be 8UC1, 8UC3 or 8UC4");
}

static void scaleImage(const cv::Mat& src, cv::Mat& dst, float scale)
{
	cv::resize(src, dst, cv::Size(cvRound(scale * src.cols), cvRound(scale * src.rows)));
}

static void scaleKeypoints(const std::vector<cv::KeyPoint>& src, std::vector<cv::KeyPoint>& dst, float scale)
{
	dst = src;
	for (auto& kpt : dst)
	{
		kpt.pt *= scale;
		kpt.size *= scale;
	}
}

void drawKeypoints(const cv::Mat& _src, const std::vector<cv::KeyPoint>& _keypoints, cv::Mat& dst, cv::Size maxSize)
{
	const float scale = std::min(1.f * maxSize.width / _src.cols, 1.f * maxSize.height / _src.rows);
	cv::Mat src;
	std::vector<cv::KeyPoint> keypoints;
	scaleImage(_src, src, scale);
	scaleKeypoints(_keypoints, keypoints, scale);
	cv::drawKeypoints(src, keypoints, dst, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

void drawMatches(const cv::Mat& _img1, const std::vector<cv::KeyPoint>& _keypoints1, const cv::Mat& _img2, const std::vector<cv::KeyPoint>& _keypoints2,
	const std::vector<cv::DMatch>& matches, cv::Mat& dst, cv::Size maxSize)
{
	const float scale = std::min(1.f * maxSize.width / _img1.cols, 1.f * maxSize.height / _img1.rows);

	cv::Mat img1, img2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	scaleImage(_img1, img1, scale);
	scaleImage(_img2, img2, scale);
	scaleKeypoints(_keypoints1, keypoints1, scale);
	scaleKeypoints(_keypoints2, keypoints2, scale);
	cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, dst);
}
