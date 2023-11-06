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
"{ @input-image    | <none> | input image.                                                       }"
"{ max-keypoints   |  10000 | maximum number of keypoints.                                       }"
"{ fast-threshold  |     20 | FAST threshold.                                                    }"
"{ num-levels      |      8 | number of pyramid levels.                                          }"
"{ nonmax-radius   |     15 | radius of non-maximum suppression.                                 }"
"{ descriptor-type |      0 | descriptor type(0:BAD 1:HashSIFT).                                 }"
"{ descriptor-bits |    256 | descriptor bits(256 or 512).                                       }"
"{ benchmark-type  |      0 | benchmark type(0:detect-and-compute 1:detect-only 2:compute-only). }"
"{ num-iterations  |    100 | number of iterations for benchmark .                               }"
"{ help  h         |        | print help message.                                                }";

template <class Function>
static double perf(int niterations, Function function)
{
	uint64_t sum = 0;
	for (int iter = 0; iter <= niterations; iter++)
	{
		const auto t0 = std::chrono::steady_clock::now();
		function();
		const auto t1 = std::chrono::steady_clock::now();
		if (iter > 0)
			sum += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	}
	return 1e-3 * sum / niterations;
}

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
	const int nlevels = parser.get<int>("num-levels");
	const int fastThreshold = parser.get<int>("fast-threshold");
	const int nonmaxRadius = parser.get<int>("nonmax-radius");
	const int descType = parser.get<int>("descriptor-type");
	const int descBits = parser.get<int>("descriptor-bits");
	const int benchType = parser.get<int>("benchmark-type");
	const int niterations = parser.get<int>("num-iterations");

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
	const char* benchStr[] = { "detect-and-compute", "detect-only", "compute-only" };

	std::cout << "=== configulations ===" << std::endl;
	std::cout << "image size      : " << image.size() << std::endl;
	std::cout << "descriptor type : " << descStr[descType] << std::endl;
	std::cout << "descriptor bits : " << descBits << std::endl;
	std::cout << "max keypoints   : " << nfeatures << std::endl;
	std::cout << "num levels      : " << nlevels << std::endl;
	std::cout << "benchmark type  : " << benchStr[benchType] << std::endl;
	std::cout << std::endl;

	cv::Mat h_gray;
	convertToGray(image, h_gray);

	// detect keypoints
	auto feature = cv::cuda::EfficientFeatures::create(nfeatures);
	feature->setNLevels(nlevels);
	feature->setFastThreshold(fastThreshold);
	feature->setNonmaxRadius(nonmaxRadius);
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	cv::cuda::GpuMat d_gray(h_gray), d_keypoints, d_descriptors;
	cv::cuda::Stream stream;

	enum { BENCHMARK_DETECT_AND_COMPUTE, BENCHMARK_DETECT_ONLY, BENCHMARK_COMPUTE_ONLY };

	double time = 0;
	if (benchType == BENCHMARK_DETECT_AND_COMPUTE)
	{
		time = perf(niterations, [&]()
		{
			feature->detectAndComputeAsync(d_gray, cv::noArray(), d_keypoints, d_descriptors, false, stream);
			stream.waitForCompletion();
		});
	}
	else if (benchType == BENCHMARK_DETECT_ONLY)
	{
		time = perf(niterations, [&]()
		{
			feature->detectAsync(d_gray, d_keypoints, cv::noArray(), stream);
			stream.waitForCompletion();
		});
	}
	else if (benchType == BENCHMARK_COMPUTE_ONLY)
	{
		feature->detectAsync(d_gray, d_keypoints, cv::noArray(), stream);
		stream.waitForCompletion();

		time = perf(niterations, [&]()
		{
			feature->computeAsync(d_gray, d_keypoints, d_descriptors, stream);
			stream.waitForCompletion();
		});
	}

	std::printf("%5d keypoints found.\n", d_keypoints.cols);
	std::printf("processing time: %.1f[milli sec]\n", time);

	return 0;
}
