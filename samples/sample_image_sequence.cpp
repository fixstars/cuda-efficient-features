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
"{ @image-format   | <none> | format of image sequence.                   }"
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
	const auto imageFormat = parser.get<std::string>("@image-format");
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

	const char* descStr[] = { "BAD", "HashSIFT" };

	std::cout << "=== configulations ===" << std::endl;
	std::cout << "descriptor type : " << descStr[descType] << std::endl;
	std::cout << "descriptor bits : " << descBits << std::endl;
	std::cout << "max keypoints   : " << nfeatures << std::endl;
	std::cout << std::endl;

	// extract features
	std::cout << "=== extract features ===" << std::endl;
	auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
	feature->setFastThreshold(fastThreshold);
	feature->setNonmaxRadius(nonmaxRadius);
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	cv::cuda::GpuMat d_image, d_keypoints, d_descriptors;
	cv::cuda::Stream stream;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	auto matcher = cv::BFMatcher::create(feature->defaultNorm());
	for (int frameId = 1; ; frameId++)
	{
		cv::Mat image = cv::imread(cv::format(imageFormat.c_str(), frameId));
		if (image.empty())
		{
			std::cout << "imread failed." << std::endl;
			break;
		}

		cv::Mat gray;
		convertToGray(image, gray);

		d_image.upload(gray);

		cv::TickMeter t;
		t.start();

		feature->detectAndComputeAsync(d_image, cv::noArray(), d_keypoints, d_descriptors, false, stream);
		stream.waitForCompletion();

		t.stop();

		feature->convert(d_keypoints, keypoints2);
		d_descriptors.download(descriptors2);

		if (descriptors1.empty())
		{
			keypoints1 = keypoints2;
			descriptors2.copyTo(descriptors1);
			continue;
		}

		std::vector<std::vector<cv::DMatch>> matches12, matches21;
		matcher->knnMatch(descriptors1, descriptors2, matches12, 2);
		matcher->knnMatch(descriptors2, descriptors1, matches21, 2);

		cv::Mat draw;
		cv::cvtColor(gray, draw, cv::COLOR_GRAY2BGR);

		const double uniqueness = 0.9;
		int nmatches = 0;
		for (const auto& m12 : matches12)
		{
			const auto& m21 = matches21[m12[0].trainIdx];

			// uniqueness check
			if (m12[0].distance > uniqueness * m12[1].distance)
				continue;

			// uniqueness check
			if (m21[0].distance > uniqueness * m21[1].distance)
				continue;

			// cross check
			if (m21[0].trainIdx != m12[0].queryIdx)
				continue;

			const auto& pt1 = keypoints1[m12[0].queryIdx].pt;
			const auto& pt2 = keypoints2[m12[0].trainIdx].pt;
			cv::line(draw, pt1, pt2, cv::Scalar(255, 255, 0));
			cv::circle(draw, pt2, 2, cv::Scalar(255, 255, 0), -1);
			nmatches++;
		}

		cv::putText(draw, cv::format("number of matches : %4d", nmatches), cv::Point(20, 20), 1, 1, cv::Scalar(0, 0, 255));
		cv::putText(draw, cv::format("compute features  : %5.1f[msec]", t.getTimeMilli()), cv::Point(20, 40), 1, 1, cv::Scalar(0, 0, 255));

		cv::imshow("matches", draw);
		const char c = cv::waitKey(1);
		if (c == 27)
			break;

		keypoints1 = keypoints2;
		descriptors2.copyTo(descriptors1);
	}

	return 0;
}
