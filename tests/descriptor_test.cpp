#include <gtest/gtest.h>

#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <efficient_descriptors.h>
#include <cuda_efficient_descriptors.h>
#include <cuda_efficient_features.h>

using Parameters = std::tuple<int, int>;

class DescriptorTest : public ::testing::TestWithParam<Parameters> {};
INSTANTIATE_TEST_CASE_P(TestWithParams, DescriptorTest,
	::testing::Combine(::testing::Values(256, 512), ::testing::Range(0, 11)));

TEST_P(DescriptorTest, BAD)
{
	const auto param = GetParam();
	const int descBits = std::get<0>(param);
	const int imageIdx = std::get<1>(param);

	const auto filename = cv::format("%s/images/100_71%02d.JPG", TEST_DATA_DIR, imageIdx);
	const int nbits = descBits == 256 ? cv::BAD::SIZE_256_BITS : cv::BAD::SIZE_512_BITS;

	auto detector = cv::cuda::EfficientFeatures::create(100000);
	auto extractorCPU = cv::BAD::create(1, nbits);
	auto extractorGPU = cv::cuda::BAD::create(1, nbits);

	cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptorsCPU, descriptorsGPU;

	detector->detect(image, keypoints);
	extractorCPU->compute(image, keypoints, descriptorsCPU);
	extractorGPU->compute(image, keypoints, descriptorsGPU);

	cv::Mat diff;
	cv::absdiff(descriptorsCPU, descriptorsGPU, diff);
	const int errors = cv::countNonZero(diff);
	const int maxErrors = cvFloor(2e-5 * descriptorsCPU.size().area());

	EXPECT_LE(errors, maxErrors);
}

TEST_P(DescriptorTest, HashSIFT)
{
	const auto param = GetParam();
	const int descBits = std::get<0>(param);
	const int imageIdx = std::get<1>(param);

	const auto filename = cv::format("%s/images/100_71%02d.JPG", TEST_DATA_DIR, imageIdx);
	const int nbits = descBits == 256 ? cv::HashSIFT::SIZE_256_BITS : cv::HashSIFT::SIZE_512_BITS;

	auto detector = cv::cuda::EfficientFeatures::create(100000);
	auto extractorCPU = cv::HashSIFT::create(1, nbits);
	auto extractorGPU = cv::cuda::HashSIFT::create(1, nbits);

	cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptorsCPU, descriptorsGPU;

	detector->detect(image, keypoints);
	extractorCPU->compute(image, keypoints, descriptorsCPU);
	extractorGPU->compute(image, keypoints, descriptorsGPU);

	cv::Mat diff;
	cv::absdiff(descriptorsCPU, descriptorsGPU, diff);
	const int errors = cv::countNonZero(diff);
	const int maxErrors = cvFloor(1e-4 * descriptorsCPU.size().area());

	EXPECT_LE(errors, maxErrors);
}
