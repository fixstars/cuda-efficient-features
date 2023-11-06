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
"{ @input-image    | <none> | input image.                                }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                }"
"{ fast-threshold  |     20 | FAST threshold.                             }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.          }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT).          }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                }"
"{ compute-async   |        | compute asynchronously.                     }"
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
	const std::string filename = parser.get<std::string>("@input-image");
	const int nfeatures = parser.get<int>("max-keypoints");
	const int fastThreshold = parser.get<int>("fast-threshold");
	const int nonmaxRadius = parser.get<int>("nonmax-radius");
	const int descType = parser.get<int>("descriptor-type");
	const int descBits = parser.get<int>("descriptor-bits");
	const bool computeAsync = parser.has("compute-async");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	cv::Mat image = cv::imread(filename);
	if (image.empty())
	{
		std::cerr << "imread failed." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const char* descStr[] = { "BAD", "HashSIFT" };

	std::cout << "=== configulations ===" << std::endl;
	std::cout << "image size      : " << image.size() << std::endl;
	std::cout << "descriptor type : " << descStr[descType] << std::endl;
	std::cout << "descriptor bits : " << descBits << std::endl;
	std::cout << "max keypoints   : " << nfeatures << std::endl;
	std::cout << "compute async   : " << (computeAsync ? "Yes" : "No") << std::endl;
	std::cout << std::endl;

	cv::Mat gray;
	convertToGray(image, gray);

	// detect keypoints
	auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
	feature->setFastThreshold(fastThreshold);
	feature->setNonmaxRadius(nonmaxRadius);
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	if (computeAsync)
	{
		cv::cuda::GpuMat d_gray(gray), d_keypoints, d_descriptors;
		cv::cuda::Stream stream;

		feature->detectAndComputeAsync(d_gray, cv::noArray(), d_keypoints, d_descriptors, false, stream);

		stream.waitForCompletion();
		feature->convert(d_keypoints, keypoints);
		d_descriptors.download(descriptors);
	}
	else
	{
		feature->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
	}

	std::cout << keypoints.size() << " keypoints found." << std::endl << std::endl;

	// draw
	cv::Mat draw;
	drawKeypoints(image, keypoints, draw);
	cv::imshow("keypoints", draw);
	cv::waitKey(0);

	return 0;
}
