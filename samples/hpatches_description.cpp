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
#include <sstream>
#include <fstream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <efficient_descriptors.h>
#include <cuda_efficient_descriptors.h>

#include "sample_common.h"

static std::string keys =
"{ @hpatchs-dir     |   <none> | path to hpatches-release.                   }"
"{ result-dir       | ./result | path to result.                             }"
"{ descriptor-type  |        0 | descriptor type(0:BAD 1:HashSIFT).          }"
"{ descriptor-bits  |      256 | descriptor bits(256 or 512).                }"
"{ compute-angle    |          | compute angles of keypoints.                }"
"{ help  h          |          | print help message.                         }";

using Path = std::filesystem::path;

static std::vector<Path> listdir(const Path& path)
{
	if (!std::filesystem::exists(path))
	{
		std::cerr << "No such directory: " << path << std::endl;
		return std::vector<Path>();
	}

	std::vector<Path> list;
	for (const auto& f : std::filesystem::directory_iterator(path))
		list.push_back(f);
	std::sort(std::begin(list), std::end(list));
	return list;
}

static std::vector<Path> findImageFiles(const Path& path)
{
	std::vector<Path> list;
	for (const auto& f : std::filesystem::directory_iterator(path))
		if (f.path().extension() == ".png")
			list.push_back(f);
	return list;
}

static void loadImages(const std::vector<Path>& filenames, std::vector<cv::Mat>& images, int flags, int nthreads = 8)
{
	const int nimages = static_cast<int>(filenames.size());

	images.resize(nimages);

#pragma omp parallel for num_threads(nthreads)
	for (int i = 0; i < nimages; i++)
		images[i] = cv::imread(filenames[i].string(), flags);
}

static void saveDescriptors(const std::string& filename, const cv::Mat& descriptors)
{
	std::ofstream ofs(filename);

	for (int i = 0; i < descriptors.rows; i++)
	{
		const uchar* ptrDesc = descriptors.ptr<uchar>(i);
		for (int j = 0; j < descriptors.cols - 1; j++)
		{
			const int d = ptrDesc[j];
			for (int k = 7; k >= 0; k--)
			{
				const int bit = (d >> k) & 0x01;
				ofs << bit << ",";
			}
		}
		{
			const int j = descriptors.cols - 1;
			const int d = ptrDesc[j];
			for (int k = 7; k >= 1; k--)
			{
				const int bit = (d >> k) & 0x01;
				ofs << bit << ",";
			}
			ofs << (d & 0x01);
		}

		ofs << std::endl;
	}
}

static void calcUMax(std::vector<int>& umax, int patchSize)
{
	// pre-compute the end of a row in a circular patch
	const int halfPatchSize = patchSize / 2;
	umax.resize(halfPatchSize + 2);

	int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
	int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
	for (v = 0; v <= vmax; ++v)
		umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

	// Make sure we are symmetric
	for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
	{
		while (umax[v0] == umax[v0 + 1])
			++v0;
		umax[v] = v0;
		++v0;
	}
}

static void ICAngles(const cv::Mat& img, std::vector<cv::KeyPoint>& pts, const std::vector<int>& u_max, int half_k)
{
	const int step = static_cast<int>(img.step1());
	const int ptsize = static_cast<int>(pts.size());

#pragma omp parallel for
	for (int ptidx = 0; ptidx < ptsize; ptidx++)
	{
		const auto& pt = pts[ptidx].pt;
		const uchar* center = &img.at<uchar>(cvFloor(pt.y), cvFloor(pt.x));

		int m_01 = 0, m_10 = 0;

		// Treat the center line differently, v=0
		for (int u = -half_k; u <= half_k; ++u)
			m_10 += u * center[u];

		// Go line by line in the circular patch
		for (int v = 1; v <= half_k; ++v)
		{
			// Proceed over the two lines
			int v_sum = 0;
			int d = u_max[v];
			for (int u = -d; u <= d; ++u)
			{
				int val_plus = center[u + v * step], val_minus = center[u - v * step];
				v_sum += (val_plus - val_minus);
				m_10 += u * (val_plus + val_minus);
			}
			m_01 += v * v_sum;
		}

		pts[ptidx].angle = cv::fastAtan2((float)m_01, (float)m_10);
	}
}

int main(int argc, char* argv[])
{
	const cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const auto hpatchesDir = parser.get<std::string>("@hpatchs-dir");
	const auto resultDir = parser.get<std::string>("result-dir");
	const int descType = parser.get<int>("descriptor-type");
	const int descBits = parser.get<int>("descriptor-bits");
	const bool computeAngle = parser.has("compute-angle");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	const char* descStr[] = { "BAD", "HashSIFT" };

	std::cout << "=== configulations ===" << std::endl;
	std::cout << "HPatchs directory : " << hpatchesDir << std::endl;
	std::cout << "result directory  : " << resultDir << std::endl;
	std::cout << "descriptor type   : " << descStr[descType] << std::endl;
	std::cout << "descriptor bits   : " << descBits << std::endl;
	std::cout << "compute angle     : " << (computeAngle ? "Yes" : "No") << std::endl;
	std::cout << std::endl;

	const auto sequenceDirs = listdir(hpatchesDir);
	const int ndirectories = static_cast<int>(sequenceDirs.size());
	std::cout << "number of patch directories: " << ndirectories << std::endl;

	constexpr int PATCH_SIZE = 65;
	constexpr int HALF_PATCH_SIZE = PATCH_SIZE / 2;

	auto feature = cv::cuda::EfficientFeatures::create();
	feature->setDescriptorType(getDescriptorType(descType, descBits));

	const auto descDir = cv::format("%s/%s_%d", resultDir.c_str(), descStr[descType], descBits);

	std::vector<int> umax;
	calcUMax(umax, PATCH_SIZE);

	int count = 0;
	for (const auto& sequenceDir : sequenceDirs)
	{
		const auto seqname = sequenceDir.filename().string();
		std::printf("sequence: %3d/%3d [%s]\n", ++count, ndirectories, seqname.c_str());

		// load images
		const auto imageFiles = findImageFiles(sequenceDir);
		std::vector<cv::Mat> images;
		loadImages(imageFiles, images, cv::IMREAD_GRAYSCALE);

		cv::Mat stacked;
		cv::hconcat(images, stacked);

		const int npatches = stacked.rows / PATCH_SIZE;
		const int nimages = static_cast<int>(images.size());

		std::cout << "patch num: " << cv::Size(npatches, nimages) << std::endl;
		std::cout << std::endl;

		// generate keypoints
		std::vector<cv::KeyPoint> keypoints;
		keypoints.reserve(npatches * nimages);

		for (int x = 0; x < nimages; x++)
		{
			for (int y = 0; y < npatches; y++)
			{
				const cv::Point2f pt(PATCH_SIZE * (x + 0.5f), PATCH_SIZE * (y + 0.5f));
				keypoints.push_back(cv::KeyPoint(pt, 64.f, -1));
			}
		}

		if (computeAngle)
			ICAngles(stacked, keypoints, umax, HALF_PATCH_SIZE);

		// compute descriptors
		cv::Mat descriptors;
		feature->compute(stacked, keypoints, descriptors);

		// save result
		const auto saveDir = cv::format("%s/%s", descDir.c_str(), seqname.c_str());
		std::filesystem::create_directories(saveDir);
		for (int x = 0; x < nimages; x++)
		{
			const auto filename = imageFiles[x].filename().replace_extension("csv").string();
			const cv::Range range(x * npatches, (x + 1) * npatches);
			saveDescriptors(saveDir + "/" + filename, descriptors.rowRange(range));
		}
	}

	return 0;
}
