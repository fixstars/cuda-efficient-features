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

#include "cuda_bad_internal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/cudev/grid/detail/integral.hpp>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{
namespace gpu
{

struct AffineParams
{
	float m00, m01, m02;
	float m10, m11, m12;
	float scale;
};

static constexpr float CV_DEGREES_TO_RADS = 0.017453292519943295f; // (M_PI / 180.0)
static constexpr float CV_BAD_EXTRA_RATIO_MARGIN = 1.75f;
static constexpr int BOX_PAIR_PARAMS_MAX_SIZE = 512;

__constant__ BoxPairParams box_pair_params[BOX_PAIR_PARAMS_MAX_SIZE];
__constant__ float thresholds_[BOX_PAIR_PARAMS_MAX_SIZE];

static __device__ inline int CV_ROUNDNUM(float x) { return (int)(x + 0.5f); }

/**
 * @brief Function that determines if a keypoint is close to the image border.
 * @param kp The detected keypoint
 * @param imgSize The size of the image
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @return true if the keypoint is in the border, false otherwise
 */
static __device__ inline bool isKeypointInTheBorder(float x, float y, float kpSize,
	int imageW, int imageH, int patchW, int patchH, float scaleFactor = 1)
{
	// This would be the correct measure but since we will compare with half of the size, use this as border size
	const float s = scaleFactor * kpSize / (patchW + patchH);

	const float borderW = patchW * s * CV_BAD_EXTRA_RATIO_MARGIN;
	const float borderH = patchH * s * CV_BAD_EXTRA_RATIO_MARGIN;

	if (x < borderW || x + borderW >= imageW)
		return true;

	if (y < borderH || y + borderH >= imageH)
		return true;

	return false;
}

/**
 * @brief Rectifies the coordinates of the box pairs measurement functions with the keypoint
 * location parameters.
 * @param boxParams The output  weak learner parameters adapted to the keypoint location
 * @param kp The keypoint defining the offset, rotation and scale to be applied
 * @param scaleFactor A scale factor that magnifies the measurement functions w.r.t. the keypoint.
 * @param patchSize The size of the normalized patch where the measurement functions were learnt.
 */
__device__ void rectifyBoxes(
	float kp_x, float kp_y, float kp_size, float kp_angle,
	float& m00, float& m01, float& m02, float& m10, float& m11, float& m12, float &scale,
	float scaleFactor = 1,
	int patch_cols = 32, int patch_rows = 32)
{
	float cosine, sine;

	scale = scaleFactor * kp_size / (0.5f * (patch_cols + patch_rows));

	if (kp_angle == -1)
	{
		m00 = scale;
		m01 = 0.0f;
		m02 = -0.5f * scale * patch_cols + kp_x;
		m10 = 0.0f;
		m11 = scale;
		m12 = -scale * 0.5f * patch_rows + kp_y;
	}
	else
	{
		cosine = (kp_angle >= 0) ? cos(kp_angle * CV_DEGREES_TO_RADS) : 1.f;
		sine = (kp_angle >= 0) ? sin(kp_angle * CV_DEGREES_TO_RADS) : 0.f;

		m00 = scale * cosine;
		m01 = -scale * sine;
		m02 = (-scale * cosine + scale * sine) * patch_cols * 0.5f + kp_x;
		m10 = scale * sine;
		m11 = scale * cosine;
		m12 = (-scale * sine - scale * cosine) * patch_rows * 0.5f + kp_y;
	}
}

static __device__ inline void calcAffineParams(float x, float y, float kpSize, float angle,
	AffineParams& M, int patchW, int patchH, float scaleFactor)
{
	const float scale = scaleFactor * kpSize / (0.5f * (patchW + patchH));
	if (angle == -1)
	{
		M.m00 = scale;
		M.m01 = 0.0f;
		M.m02 = -0.5f * scale * patchW + x;
		M.m10 = 0.0f;
		M.m11 = scale;
		M.m12 = -scale * 0.5f * patchH + y;
	}
	else
	{
		const float cost = (angle >= 0) ? float(cos(angle * CV_DEGREES_TO_RADS)) : 1.f;
		const float sint = (angle >= 0) ? float(sin(angle * CV_DEGREES_TO_RADS)) : 0.f;

		M.m00 = scale * cost;
		M.m01 = -scale * sint;
		M.m02 = (-scale * cost + scale * sint) * patchW * 0.5f + x;
		M.m10 = scale * sint;
		M.m11 = scale * cost;
		M.m12 = (-scale * sint - scale * cost) * patchH * 0.5f + y;
	}
	M.scale = scale;
}

static __device__ inline void transformBoxPairParams(const BoxPairParams& src, BoxPairParams& dst, const AffineParams& M)
{
	dst.x1 = CV_ROUNDNUM(M.m00 * src.x1 + M.m01 * src.y1 + M.m02);
	dst.y1 = CV_ROUNDNUM(M.m10 * src.x1 + M.m11 * src.y1 + M.m12);
	dst.x2 = CV_ROUNDNUM(M.m00 * src.x2 + M.m01 * src.y2 + M.m02);
	dst.y2 = CV_ROUNDNUM(M.m10 * src.x2 + M.m11 * src.y2 + M.m12);
	dst.boxRadius = CV_ROUNDNUM(M.scale * src.boxRadius);
	// dst.th = src.th;
}

/**
 * @brief Computes the Box Average Difference, measuring the difference of gray level in the two
 * square regions.
 * @param boxParams The box parameter defining the size and locations of each box.
 * @param integralImage The integral image used to compute the average gray value in the square regions.
 * @return The difference of gray level in the two squares defined by box_params
 */
__device__ float computeBadResponse(const BoxPairParams& boxParams, const PtrStepSz<int> integralImage)
{
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

	// Read the integral image values for the first box
	A = integralImage(box1y1, box1x1);
	B = integralImage(box1y1, box1x2);
	C = integralImage(box1y2, box1x1);
	D = integralImage(box1y2, box1x2);

	// Calculate the mean intensity value of the pixels in the box
	sum1 = float(A + D - B - C);
	box_area1 = (box1y2 - box1y1) * (box1x2 - box1x1);
	average1 = sum1 / box_area1;

	// Calculate the indices on the integral image where the box falls
	A = integralImage(box2y1, box2x1);
	B = integralImage(box2y1, box2x2);
	C = integralImage(box2y2, box2x1);
	D = integralImage(box2y2, box2x2);

	// Calculate the mean intensity value of the pixels in the box
	sum2 = float(A + D - B - C);
	box_area2 = (box2y2 - box2y1) * (box2x2 - box2x1);
	average2 = sum2 / box_area2;

	return average1 - average2;
}


__global__ void computeBADKernel(const PtrStepSzi integral, const float4* keypoints, int nkeypoints,
	PtrStep<uchar> descriptors, float scaleFactor, int paramSize, int patchW, int patchH)
{
	const int kpIdx = blockDim.y * blockIdx.y + threadIdx.y;
	const int boxIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int bitIdx = 7 - (boxIdx % 8);

	const int frameW = integral.cols - 1;
	const int frameH = integral.rows - 1;

	const float4 kpt = keypoints[kpIdx];
	const float x = kpt.x;
	const float y = kpt.y;
	const float kpSize = kpt.z;
	const float angle = kpt.w;

	uchar byte = 0;
	BoxPairParams box_pair;
	AffineParams M;

	if (kpIdx < nkeypoints)
	{
		calcAffineParams(x, y, kpSize, angle, M, patchW, patchH, scaleFactor);
		transformBoxPairParams(box_pair_params[boxIdx], box_pair, M);

		if (isKeypointInTheBorder(x, y, kpSize, frameW, frameH, patchW, patchH, scaleFactor))
		{
			const float responseFun = computeBadResponse(box_pair, integral);
			// Set the bit to 1 if the response function is less or equal to the threshod
			byte |= (responseFun <= thresholds_[boxIdx]) << bitIdx;
		}
		else
		{
			// For the first box, we calculate its margin coordinates
			const int box1x1 = box_pair.x1 - box_pair.boxRadius;
			const int box1y1 = box_pair.y1 - box_pair.boxRadius;
			const int box1x2 = box_pair.x1 + box_pair.boxRadius + 1;
			const int box1y2 = box_pair.y1 + box_pair.boxRadius + 1;
			// For the second box, we calculate its margin coordinates
			const int box2x1 = box_pair.x2 - box_pair.boxRadius;
			const int box2y1 = box_pair.y2 - box_pair.boxRadius;
			const int box2x2 = box_pair.x2 + box_pair.boxRadius + 1;
			const int box2y2 = box_pair.y2 + box_pair.boxRadius + 1;
			const int side = 1 + (box_pair.boxRadius << 1);

			// Get the difference between the average level of the two boxes
			const int areaResponseFun = (
				integral(box1y1, box1x1)               // A of Box1
				+ integral(box1y2, box1x2)               // D of Box1
				- integral(box1y1, box1x2)               // B of Box1
				- integral(box1y2, box1x1)               // C of Box1
				- integral(box2y1, box2x1)               // A of Box2
				- integral(box2y2, box2x2)               // D of Box2
				+ integral(box2y1, box2x2)               // B of Box2
				+ integral(box2y2, box2x1));             // C of Box2

			// Set the bit to 1 if the response function is less or equal to the threshod
			byte |= (areaResponseFun <= (thresholds_[boxIdx] * (side * side))) << bitIdx;
		}  // End of else (of pixels in the image center)

		byte |= __shfl_xor_sync(0xffffffff, byte, 4);
		byte |= __shfl_xor_sync(0xffffffff, byte, 2);
		byte |= __shfl_xor_sync(0xffffffff, byte, 1);

		if (bitIdx == 0)
		{
			const int byteIdx = boxIdx / 8;
			descriptors(kpIdx, byteIdx) = byte;
		}
	}
}

void loadBoxPairParams(int paramSIze)
{
#include "bad.p512.h"
#include "bad.p256.h"
	if (paramSIze == 512)
	{
		CUDA_CHECK(cudaMemcpyToSymbol(box_pair_params, box_pair_params_512, 512 * sizeof(box_pair_params_512[0])));
		CUDA_CHECK(cudaMemcpyToSymbol(thresholds_, thresholds_512, 512 * sizeof(thresholds_512[0])));
	}
	else if (paramSIze == 256)
	{
		CUDA_CHECK(cudaMemcpyToSymbol(box_pair_params, box_pair_params_256, 256 * sizeof(box_pair_params_256[0])));
		CUDA_CHECK(cudaMemcpyToSymbol(thresholds_, thresholds_256, 256 * sizeof(thresholds_256[0])));
	}
	else
		CV_Error(Error::StsBadArg, "n_boxes should be either SIZE_512_BITS or SIZE_256_BITS");
}

void computeBAD(const GpuMat& integral, const GpuMat& keypoints, GpuMat& descriptors,
	float scaleFactor, int paramSize, Size patchSize, cudaStream_t stream)
{
	const int nkeypoints = keypoints.rows;
	const int BLOCK_SIZE_X = 16;
	const int BLOCK_SIZE_Y = 16;
	const dim3 dimGrid(divUp(paramSize, BLOCK_SIZE_X), divUp(nkeypoints, BLOCK_SIZE_Y), 1);
	const dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

	computeBADKernel<<<dimGrid, dimBlock, 0, stream>>>(integral, keypoints.ptr<float4>(), nkeypoints, descriptors,
		scaleFactor, paramSize, patchSize.width, patchSize.height);
	CUDA_CHECK(cudaGetLastError());
}

void calcIntegralImage(const GpuMat& src, GpuMat& dst, Stream& stream)
{
	using namespace cudev;

	//CV_Assert(dst.rows == src.rows + 1 && dst.cols == src.cols + 1 && dst.type() == CV_32S);
	const int rows = src.rows;
	const int cols = src.cols;

	dst.create(rows + 1, cols + 1, CV_32S);
	dst.setTo(0, stream);

	GpuMat dstROI = dst(Rect(1, 1, src.cols, src.rows));
	integral_detail::integral(globPtr<uchar>(src), globPtr<int>(dstROI), rows, cols, StreamAccessor::getStream(stream));
}

} // namespace gpu
} // namespace cuda
} // namespace cv
