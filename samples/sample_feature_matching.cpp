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

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_efficient_features.h>

#include "sample_common.h"

static std::string keys =
"{ @first-image    | <none> | first input image.                          }"
"{ @second-image   | <none> | second input image.                         }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                }"
"{ fast-threshold  |     20 | FAST threshold.                             }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.          }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT).          }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                }"
"{ help  h         |        | print help message.                         }";

int main(int argc, char* argv[])
{
	const cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	// get parameters
	const std::string filename1 = parser.get<std::string>("@first-image");
	const std::string filename2 = parser.get<std::string>("@second-image");
	const int nfeatures = parser.get<int>("max-keypoints");
	const int fastThreshold = parser.get<int>("fast-threshold");
	const int nonmaxRadius = parser.get<int>("nonmax-radius");
	const int descType = parser.get<int>("descriptor-type");
	const int descBits = parser.get<int>("descriptor-bits");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	cv::Mat image1 = cv::imread(filename1);
	cv::Mat image2 = cv::imread(filename2);
	if (image1.empty() || image2.empty())
	{
		std::cerr << "imread failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const char* descStr[] = { "BAD", "HashSIFT" };

	std::cout << "=== configulations ===" << std::endl;
	std::cout << "image size      : " << image1.size() << " and " << image2.size() << std::endl;
	std::cout << "descriptor type : " << descStr[descType] << std::endl;
	std::cout << "descriptor bits : " << descBits << std::endl;
	std::cout << "max keypoints   : " << nfeatures << std::endl;
	std::cout << std::endl;

	cv::Mat gray1, gray2;
	convertToGray(image1, gray1);
	convertToGray(image2, gray2);

	// extract features
	std::cout << "=== extract features ===" << std::endl;
	auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	feature->setFastThreshold(fastThreshold);
	feature->setNonmaxRadius(nonmaxRadius);
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	feature->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
	feature->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

	std::cout << "number of keypoins: " << keypoints1.size() << " " << keypoints2.size() << std::endl;

	// match features
	std::cout << "=== match features ===" << std::endl;
	auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
	std::vector<cv::DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);

	std::cout << "number of matches: " << matches.size() << std::endl;

	// draw
	cv::Mat draw;
	drawMatches(image1, keypoints1, image2, keypoints2, matches, draw);
	cv::imshow("image", draw);
	cv::waitKey(0);

	return 0;
}
