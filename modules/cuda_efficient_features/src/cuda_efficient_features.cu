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

#include "cuda_efficient_features.h"

#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{

static constexpr int CELL_SIZE = 16;
static constexpr int PATCH_SIZE = 31;
static constexpr int HALF_PATCH_SIZE = 15;
static constexpr float HARRIS_K = 0.04f;

static constexpr int LOCATION_ROW = EfficientFeatures::LOCATION_ROW;
static constexpr int RESPONSE_ROW = EfficientFeatures::RESPONSE_ROW;
static constexpr int ANGLE_ROW    = EfficientFeatures::ANGLE_ROW;
static constexpr int OCTAVE_ROW   = EfficientFeatures::OCTAVE_ROW;
static constexpr int SIZE_ROW     = EfficientFeatures::SIZE_ROW;
static constexpr int ROWS_COUNT   = EfficientFeatures::ROWS_COUNT;

static __device__ inline int distanceSq(short2 pt1, short2 pt2)
{
	const int dx = pt1.x - pt2.x;
	const int dy = pt1.y - pt2.y;
	return dx * dx + dy * dy;
}

static __device__ inline float convertToDegree(float angle)
{
	constexpr float PI = static_cast<float>(CV_PI);
	if (angle < 0)
		angle += 2.f * PI;
	return (180.f / PI) * angle;
}

static __device__ inline bool IsMaxPoint(int idx1, const short2* points, const float* responses,
	const int* blockPtr, const int* pointIds, int gridW, int gridH, int imageRadius, int blockRadius)
{
	const short2 pt1 = points[idx1];
	const float response1 = responses[idx1];

	const int bx1 = pt1.x / CELL_SIZE;
	const int by1 = pt1.y / CELL_SIZE;

	const int minx = ::max(bx1 - blockRadius, 0);
	const int maxx = ::min(bx1 + blockRadius, gridW - 1);
	const int miny = ::max(by1 - blockRadius, 0);
	const int maxy = ::min(by1 + blockRadius, gridH - 1);

	for (int by = miny; by <= maxy; by++)
	{
		for (int bx = minx; bx <= maxx; bx++)
		{
			const int blockId = by * gridW + bx;
			for (int k = blockPtr[blockId]; k < blockPtr[blockId + 1]; k++)
			{
				const int idx2 = pointIds[k];
				if (idx1 == idx2)
					continue;

				const short2 pt2 = points[idx2];
				const float response2 = responses[idx2];

				if (response1 <= response2 && distanceSq(pt1, pt2) < imageRadius)
					return false;
			}
		}
	}

	return true;
};

static __device__ float calcResponse(PtrStepb image, short2 pt)
{
	constexpr int BLOCK_SIZE = 7;
	constexpr int RADIUS = BLOCK_SIZE / 2;
	constexpr float SCALE = 1.f / (4 * BLOCK_SIZE * 255);

	const int x0 = pt.x;
	const int y0 = pt.y;

	float sxx = 0, sxy = 0, syy = 0;
	for (int iy = -RADIUS; iy <= RADIUS; ++iy)
	{
		for (int ix = -RADIUS; ix <= RADIUS; ++ix)
		{
			const int x = x0 + ix;
			const int y = y0 + iy;

			const int v00 = image(y - 1, x - 1);
			const int v01 = image(y - 1, x);
			const int v02 = image(y - 1, x + 1);

			const int v10 = image(y, x - 1);
			const int v12 = image(y, x + 1);

			const int v20 = image(y + 1, x - 1);
			const int v21 = image(y + 1, x);
			const int v22 = image(y + 1, x + 1);

			const float dx = SCALE * ((v02 + 2 * v12 + v22) - (v00 + 2 * v10 + v20));
			const float dy = SCALE * ((v20 + 2 * v21 + v22) - (v00 + 2 * v01 + v02));
			sxx += dx * dx;
			sxy += dx * dy;
			syy += dy * dy;
		}
	}

	const float detM = sxx * syy - sxy * sxy;
	const float trM = sxx + syy;

	return detM - HARRIS_K * trM * trM;
}

static __device__ float IC_Angle(PtrStepb image, short2 pt)
{
	constexpr int U_MAX[] = { 15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3, 0 };

	const int x = pt.x;
	const int y = pt.y;

	int m_01 = 0, m_10 = 0;

	// Treat the center line differently, v=0
	for (int dx = -HALF_PATCH_SIZE; dx <= HALF_PATCH_SIZE; ++dx)
		m_10 += dx * image(y, x + dx);

	// Go line by line in the circuI853lar patch
	for (int dy = 1; dy <= HALF_PATCH_SIZE; ++dy)
	{
		// Proceed over the two lines
		int y_sum = 0;
		const int d = U_MAX[dy];
		for (int dx = -d; dx <= d; ++dx)
		{
			const int valT = image(y - dy, x + dx);
			const int valB = image(y + dy, x + dx);

			y_sum += (valB - valT);
			m_10 += dx * (valB + valT);
		}
		m_01 += dy * y_sum;
	}

	return convertToDegree(::atan2f((float)m_01, (float)m_10));
}

__global__ void nptPerBlockKernel(const short2* points, int npoints, int* nptPerBlock, int gridStep)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	const short2 pt = points[i];
	const int bx = pt.x / CELL_SIZE;
	const int by = pt.y / CELL_SIZE;
	const int blockId = by * gridStep + bx;
	atomicAdd(&nptPerBlock[blockId], 1);
}

__global__ void assignIndexKernel(const short2* points, int npoints, int* pointIds, int* nptPerBlock, int gridStep)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	const short2 pt = points[i];
	const int cx = pt.x / CELL_SIZE;
	const int cy = pt.y / CELL_SIZE;
	const int blockId = cy * gridStep + cx;

	const int k = atomicAdd(&nptPerBlock[blockId], 1);
	pointIds[k] = i;
}

__global__ void radiusSuppressionKernel(const short2* srcPts, const float* srcRes, int npoints,
	short2* dstPts, float* dstRes, int* count, const int* blockPtr, const int* pointIds,
	int gridW, int gridH, int imageRadius, int blockRadius)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	if (IsMaxPoint(i, srcPts, srcRes, blockPtr, pointIds, gridW, gridH, imageRadius, blockRadius))
	{
		const int k = atomicAdd(count, 1);
		dstPts[k] = srcPts[i];
		dstRes[k] = srcRes[i];
	}
}

__global__ void calcResponsesKernel(PtrStepb image, const short2* points, float* responses, int npoints)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	responses[i] = calcResponse(image, points[i]);
}

__global__ void calcAnglesKernel(PtrStepb image, const short2* points, float* angles, int npoints)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	angles[i] = IC_Angle(image, points[i]);
}

__global__ void scalePointsKernel(short2* points, int* octaves, float* sizes, int npoints, float scale, int octave)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	short2 pt = points[i];
	pt.x = static_cast<short>(scale * pt.x + 0.5f);
	pt.y = static_cast<short>(scale * pt.y + 0.5f);
	points[i] = pt;
	octaves[i] = octave;
	sizes[i] = scale * PATCH_SIZE;
}

__global__ void convertKeypointsKernel(const short2* srcLoc, const float* srcAngles, float4* dstKeypoints, int npoints)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= npoints)
		return;

	const short2 pt = srcLoc[i];
	float4 kpt;
	kpt.x = pt.x;
	kpt.y = pt.y;
	kpt.z = PATCH_SIZE;
	kpt.w = srcAngles[i];
	dstKeypoints[i] = kpt;
}

static void exclusiveScan(const int* src, int* dst, int size, cudaStream_t stream = 0)
{
	auto ptrSrc = thrust::device_pointer_cast(src);
	auto ptrDst = thrust::device_pointer_cast(dst);
	thrust::exclusive_scan(thrust::cuda::par.on(stream), ptrSrc, ptrSrc + size, ptrDst);
}

int radiusSuppressionBufferSize(Size imgSize, int npoints)
{
	const int gridW = divUp(imgSize.width, CELL_SIZE);
	const int gridH = divUp(imgSize.height, CELL_SIZE);
	const int nblocks = gridW * gridH;
	const int ptrSize = nblocks + 1;
	return 2 * ptrSize + npoints;
}

void radiusSuppression(const GpuMat& src, GpuMat& dst, Size imgSize, float radius,
	GpuMat& d_buffer, HostMem& h_buffer, cudaStream_t stream)
{
	const int npoints = src.cols;
	if (npoints <= 0) {
		dst.cols = 0;
		return;
	}

	const int imageRadius = cvCeil(radius * radius);
	const int blockRadius = cvCeil(radius / CELL_SIZE);

	struct Config { int block, grid; } cfg;
	cfg.block = 512;
	cfg.grid = divUp(npoints, cfg.block);

	const int gridW = divUp(imgSize.width, CELL_SIZE);
	const int gridH = divUp(imgSize.height, CELL_SIZE);
	const int nblocks = gridW * gridH;
	const int ptrSize = nblocks + 1;

	CV_Assert(dst.rows >= ROWS_COUNT && dst.cols >= npoints && dst.type() == CV_32F);
	CV_Assert(d_buffer.size().area() >= 2 * ptrSize + npoints);
	CV_Assert(h_buffer.size().area() >= 1);

	const short2* srcPts = src.ptr<short2>(LOCATION_ROW);
	const float* srcRes = src.ptr<float>(RESPONSE_ROW);
	short2* dstPts = dst.ptr<short2>(LOCATION_ROW);
	float* dstRes = dst.ptr<float>(RESPONSE_ROW);

	int* nptPerBlock = d_buffer.ptr<int>();
	int* blockPtr = d_buffer.ptr<int>() + 1 * ptrSize;
	int* pointIds = d_buffer.ptr<int>() + 2 * ptrSize;
	int* d_count = nptPerBlock + nblocks;
	int* h_count = h_buffer.createMatHeader().ptr<int>();

	// count number of points per block
	CUDA_CHECK(cudaMemsetAsync(nptPerBlock, 0, sizeof(int) * ptrSize, stream));
	nptPerBlockKernel<<<cfg.grid, cfg.block, 0, stream>>>(srcPts, npoints, nptPerBlock, gridW);
	CUDA_CHECK(cudaGetLastError());

	// calculate start addresses corresponding to each blocks
	exclusiveScan(nptPerBlock, blockPtr, ptrSize, stream);

	// assign point indices to blocks
	CUDA_CHECK(cudaMemcpyAsync(nptPerBlock, blockPtr, sizeof(int) * nblocks, cudaMemcpyDeviceToDevice, stream));
	assignIndexKernel<<<cfg.grid, cfg.block, 0, stream>>>(srcPts, npoints, pointIds, nptPerBlock, gridW);
	CUDA_CHECK(cudaGetLastError());

	// radius suppression
	radiusSuppressionKernel<<<cfg.grid, cfg.block, 0, stream>>>(srcPts, srcRes, npoints, dstPts, dstRes, d_count,
		blockPtr, pointIds, gridW, gridH, imageRadius, blockRadius);
	CUDA_CHECK(cudaGetLastError());

	// get number of remaining points 
	CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	dst.cols = *h_count;
}

void limitPoints(GpuMat& points, int maxpoints, cudaStream_t stream)
{
	const int npoints = points.cols;
	if (npoints <= maxpoints)
		return;

	auto locations = thrust::device_pointer_cast(points.ptr<short2>(0));
	auto responses = thrust::device_pointer_cast(points.ptr<float>(1));

	thrust::sort_by_key(thrust::cuda::par.on(stream), responses, responses + npoints, locations, thrust::greater<float>());

	points.cols = maxpoints;

	CUDA_CHECK(cudaGetLastError());
}

void calcResponses(const GpuMat& image, GpuMat& points, cudaStream_t stream)
{
	const int npoints = points.cols;
	if (npoints <= 0)
		return;

	const int block = 512;
	const int grid = divUp(npoints, block);

	const short2* locations = points.ptr<short2>(LOCATION_ROW);
	float* responses = points.ptr<float>(RESPONSE_ROW);

	calcResponsesKernel<<<grid, block, 0, stream>>>(image, locations, responses, npoints);
	CUDA_CHECK(cudaGetLastError());
}

void calcAngles(const GpuMat& image, GpuMat& points, cudaStream_t stream)
{
	const int npoints = points.cols;
	if (npoints <= 0)
		return;

	const int block = 512;
	const int grid = divUp(npoints, block);

	const short2* locations = points.ptr<short2>(LOCATION_ROW);
	float* angles = points.ptr<float>(ANGLE_ROW);

	calcAnglesKernel<<<grid, block, 0, stream>>>(image, locations, angles, npoints);
	CUDA_CHECK(cudaGetLastError());
}

void scalePoints(GpuMat& points, float scale, int octave, cudaStream_t stream)
{
	const int npoints = points.cols;
	if (npoints <= 0)
		return;

	const int block = 512;
	const int grid = divUp(npoints, block);

	short2* locations = points.ptr<short2>(LOCATION_ROW);
	int* octaves = points.ptr<int>(OCTAVE_ROW);
	float* sizes = points.ptr<float>(SIZE_ROW);

	scalePointsKernel<<<grid, block, 0, stream>>>(locations, octaves, sizes, npoints, scale, octave);
	CUDA_CHECK(cudaGetLastError());
}

void convertKeypoints(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
{
	const int npoints = src.cols;
	if (npoints <= 0)
	{
		dst.release();
		return;
	}

	const int block = 512;
	const int grid = divUp(npoints, block);

	CV_Assert(dst.rows >= npoints && dst.cols >= 1 && dst.type() == CV_32FC4);

	const short2* srcLoc = src.ptr<short2>(LOCATION_ROW);
	const float* srcAngles = src.ptr<float>(ANGLE_ROW);
	float4* dstKeypoints = dst.ptr<float4>(0);

	convertKeypointsKernel<<<grid, block, 0, stream>>>(srcLoc, srcAngles, dstKeypoints, npoints);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace cv
