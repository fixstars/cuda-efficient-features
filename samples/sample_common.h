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

#ifndef __SAMPLE_COMMON_H__
#define __SAMPLE_COMMON_H__

#include <opencv2/core.hpp>
#include <cuda_efficient_features.h>

enum { BAD, HashSIFT };

cv::cuda::EfficientFeatures::DescriptorType getDescriptorType(int descType, int descBits);
void convertToGray(const cv::Mat& src, cv::Mat& dst);
void drawKeypoints(const cv::Mat& src, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& dst, cv::Size maxSize = cv::Size(2048, 1024));
void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
	const std::vector<cv::DMatch>& matches, cv::Mat& dst, cv::Size maxSize = cv::Size(2048, 1024));

#endif // !__SAMPLE_COMMON_H__
