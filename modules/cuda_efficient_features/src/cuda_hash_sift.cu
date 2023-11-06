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

#include "cuda_hash_sift_internal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{
namespace gpu
{

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr int WARP_SIZE = 32;
static constexpr int DESCRIPTOR_SIZE = 128;

static constexpr float PI_1_0F = static_cast<float>(CV_PI);
static constexpr float PI_2_0F = static_cast<float>(CV_2PI);

// assumed gaussian blur for input image
static constexpr double SIFT_INIT_SIGMA = 0.5;

// determines the size of a single descriptor orientation histogram
static constexpr float SIFT_DESCR_SCL_FCTR = 3.f;

static constexpr int R_BINS = 4; // width of descriptor histogram array
static constexpr int C_BINS = 4; // width of descriptor histogram array
static constexpr int ORI_BINS = 8; // number of bins per histogram in descriptor array
static constexpr float MAGNITUDE_TH = 0.2f;
static constexpr float INT_DESCR_FACTOR = 512.f;

static constexpr int PATCH_H = 32;
static constexpr int PATCH_W = 32;

static constexpr int SIFT_BLOCK_SIZE_X = PATCH_W;
static constexpr int SIFT_BLOCK_SIZE_Y = 4;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Type definitions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct KeyPoint
{
	float x, y, size, angle;
};

struct Matx23f
{
	__device__ inline float& operator()(int i, int j) { return val[i][j]; }
	__device__ inline float operator()(int i, int j) const { return val[i][j]; }
	float val[2][3];
};

using HistPtrT = float(*)[C_BINS + 2][ORI_BINS + 2];

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static __device__ inline int floorToInt(float x)
{
	return static_cast<int>(floor(x));
}

__device__ static inline float lerp(float v0, float v1, float t)
{
	return fmaf(t, v1, fmaf(-t, v0, v0));
}

static __device__ inline int threadIdx1D()
{
	return threadIdx.y * blockDim.x + threadIdx.x;
}

static __device__ inline int blockSize1D()
{
	return blockDim.x * blockDim.y;
}

static __device__ void warpAffineLinear(const PtrStepSzb image, uchar patch[PATCH_W][PATCH_W], const Matx23f& M)
{
	const float M00 = M(0, 0);
	const float M01 = M(0, 1);
	const float M02 = M(0, 2);
	const float M10 = M(1, 0);
	const float M11 = M(1, 1);
	const float M12 = M(1, 2);

	for (int y = threadIdx.y; y < PATCH_H; y += blockDim.y)
	{
		for (int x = threadIdx.x; x < PATCH_W; x += blockDim.x)
		{
			const float u = M00 * x + M01 * y + M02;
			const float v = M10 * x + M11 * y + M12;

			uchar dstVal = 0;
			const int ui = floorToInt(u);
			const int vi = floorToInt(v);
			if (ui >= 0 && ui + 1 < image.cols && vi >= 0 && vi + 1 < image.rows)
			{
				const float du = u - ui;
				const float dv = v - vi;
				const float tmp0 = lerp(image(vi + 0, ui + 0), image(vi + 0, ui + 1), du);
				const float tmp1 = lerp(image(vi + 1, ui + 0), image(vi + 1, ui + 1), du);
				const float tmp2 = lerp(tmp0, tmp1, dv);
				dstVal = static_cast<uchar>(::min(static_cast<int>(tmp2 + 0.5f), 255));
			}
			patch[y][x] = dstVal;
		}
	}
}

static __device__ inline Matx23f getAffineTransform(const KeyPoint& kpt, float scaleFactor)
{
	const float s = scaleFactor * kpt.size / (0.5f * (PATCH_W + PATCH_H));
	const float theta = PI_1_0F * kpt.angle / 180;

	const float cost = s * (kpt.angle >= 0 ? cosf(theta) : 1.f);
	const float sint = s * (kpt.angle >= 0 ? sinf(theta) : 0.f);

	Matx23f M;

	M(0, 0) = +cost;
	M(0, 1) = -sint;
	M(0, 2) = (-cost + sint) * PATCH_W / 2.f + kpt.x;

	M(1, 0) = +sint;
	M(1, 1) = +cost;
	M(1, 2) = (-sint - cost) * PATCH_H / 2.f + kpt.y;

	return M;
}

static __device__ void rectifyPatch(const PtrStepSzb image, const KeyPoint& kpt, uchar patch[PATCH_H][PATCH_W], float scaleFactor)
{
	warpAffineLinear(image, patch, getAffineTransform(kpt, scaleFactor));
}

static __device__ inline uchar clip(float x)
{
	if (x < 0)
		return 0;
	if (x > 255)
		return 255;
	return static_cast<uchar>(x + 0.5);
}

static __device__ inline float squared(float x)
{
	return x * x;
}

static __device__ inline float normsq(float x, float y)
{
	return squared(x) + squared(y);
}

static __device__ void normalizeDescriptors(float* descriptors)
{
	if (threadIdx.y != 0)
		return;

	float sum = 0.f;
	for (int i = threadIdx.x; i < DESCRIPTOR_SIZE; i += WARP_SIZE)
		sum += squared(descriptors[i]);

	for (int mask = 16; mask > 0; mask /= 2)
		sum += __shfl_xor_sync(0xffffffff, sum, mask);

	const float norm = ::max(sqrtf(sum), FLT_EPSILON);
	const float scale = 1.f / norm;
	for (int i = threadIdx.x; i < DESCRIPTOR_SIZE; i += WARP_SIZE)
		descriptors[i] = scale * descriptors[i];
}

static __device__ inline void separateIF(float value, int* vi, float* vf)
{
	*vi = floorToInt(value);
	*vf = value - *vi;
}

static __device__ inline void distribute(float value, float weight, float* v0, float* v1)
{
	*v1 = weight * value;
	*v0 = value - *v1;
}

struct HistBin
{
	__device__ HistBin(int h, int w, float kpScale)
	{
		const float cellh = SIFT_DESCR_SCL_FCTR * (kpScale * h * 0.5f);
		const float cellw = SIFT_DESCR_SCL_FCTR * (kpScale * w * 0.5f);

		scaleR = 1.f / cellh;
		scaleC = 1.f / cellw;
		scaleO = ORI_BINS / PI_2_0F;

		halfh = 0.5f * h;
		halfw = 0.5f * w;
		rbin0 = R_BINS / 2 - 0.5f;
		cbin0 = C_BINS / 2 - 0.5f;
	}

	__device__ inline float getRBin(int r) const { return scaleR * (r - halfh) + rbin0; }
	__device__ inline float getCBin(int c) const { return scaleC * (c - halfw) + cbin0; }
	__device__ inline float getOBin(int o) const { return scaleO * o; }

	float scaleR, scaleC, scaleO, halfh, halfw, rbin0, cbin0;
};

struct Histogram
{
	__device__ inline Histogram(HistPtrT hist, const HistBin& bin) : hist(hist), bin(bin) {}

	__device__ inline void clear()
	{
		for (int r = threadIdx.y; r < R_BINS + 2; r += blockDim.y)
			for (int c = threadIdx.x; c < C_BINS + 2; c += blockDim.x)
				for (int o = 0; o < ORI_BINS + 2; o++)
					hist[r][c][o] = 0.f;
	}

	__device__ inline void vote(int x, int y, float ori, float mag)
	{
		int ri;
		float rf;
		separateIF(bin.getRBin(y), &ri, &rf);

		int ci;
		float cf;
		separateIF(bin.getCBin(x), &ci, &cf);

		int oi;
		float of;
		separateIF(bin.scaleO * ori, &oi, &of);

		if (oi < 0)
			oi += ORI_BINS;
		if (oi >= ORI_BINS)
			oi -= ORI_BINS;

		// distribute along r
		float v0, v1;
		distribute(mag, rf, &v0, &v1);

		// distribute along c
		float v00, v01, v10, v11;
		distribute(v0, cf, &v00, &v01);
		distribute(v1, cf, &v10, &v11);

		// distribute along o
		float v000, v001, v010, v011, v100, v101, v110, v111;
		distribute(v00, of, &v000, &v001);
		distribute(v01, of, &v010, &v011);
		distribute(v10, of, &v100, &v101);
		distribute(v11, of, &v110, &v111);

		atomicAdd(&hist[ri + 1][ci + 1][oi + 0], v000);
		atomicAdd(&hist[ri + 1][ci + 1][oi + 1], v001);
		atomicAdd(&hist[ri + 1][ci + 2][oi + 0], v010);
		atomicAdd(&hist[ri + 1][ci + 2][oi + 1], v011);
		atomicAdd(&hist[ri + 2][ci + 1][oi + 0], v100);
		atomicAdd(&hist[ri + 2][ci + 1][oi + 1], v101);
		atomicAdd(&hist[ri + 2][ci + 2][oi + 0], v110);
		atomicAdd(&hist[ri + 2][ci + 2][oi + 1], v111);
	}

	__device__ inline void finalize(float* descriptors)
	{
		for (int r = threadIdx.y; r < R_BINS; r += blockDim.y)
		{
			for (int c = threadIdx.x; c < C_BINS; c += blockDim.x)
			{
				// Increase the value in the penultimate orientation bin in the first one
				hist[r + 1][c + 1][0] += hist[r + 1][c + 1][ORI_BINS + 0];

				// Increase the value in last orientation bin in the second one
				hist[r + 1][c + 1][1] += hist[r + 1][c + 1][ORI_BINS + 1];

				// Copy the values in the histogram to the output destination
				for (int k = 0; k < ORI_BINS; k++)
					descriptors[(r * R_BINS + c) * ORI_BINS + k] = hist[r + 1][c + 1][k];
			}
		}
	}

	HistPtrT hist;
	HistBin bin;
};

static __device__ void createHistogram(const uchar patch[PATCH_H][PATCH_W], Histogram& hist, float kpScale)
{
	const int h = PATCH_H;
	const int w = PATCH_W;

	const float kpRadius = kpScale * h * 0.5f;
	const float kernelSigma = 0.5f * C_BINS * SIFT_DESCR_SCL_FCTR * kpRadius;
	const float distScale = -1.f / (2 * kernelSigma * kernelSigma);
	const float cx = 0.5f * w;
	const float cy = 0.5f * h;

	constexpr int CELL_SIZE = 8;
	constexpr int STRIDE_X = 4;
	constexpr int STRIDE_Y = CELL_SIZE / SIFT_BLOCK_SIZE_Y;

	const int niterations = PATCH_W * PATCH_H / (SIFT_BLOCK_SIZE_X * SIFT_BLOCK_SIZE_Y);
	const int tid = threadIdx1D();
	for (int iter = 0; iter < niterations; iter++)
	{
		const int y = (tid / CELL_SIZE) * STRIDE_Y + (iter / STRIDE_X);
		const int x = (tid % CELL_SIZE) * STRIDE_X + (iter % STRIDE_X);

		if (y >= 1 && y < h - 1 && x >= 1 && x < w - 1)
		{
			// Multiply the gradient magnitude by the importance of each pixel
			const float magScale = expf(distScale * normsq(x - cx, y - cy));

			// Compute the derivative using the previous and next pixels
			const float dx = patch[y + 0][x + 1] - patch[y + 0][x - 1];
			const float dy = patch[y - 1][x + 0] - patch[y + 1][x + 0];
			const float mag = magScale * sqrtf(normsq(dx, dy));
			const float ori = atan2f(dy, dx);

			hist.vote(x, y, ori, mag);
		}
	}
}

static __device__ void describeFeatureVector(Histogram& hist, float* descriptors)
{
	hist.finalize(descriptors);
	__syncthreads();

	// Step 7: Apply L2 normalization
	normalizeDescriptors(descriptors);
	__syncthreads();

	// Step 8: Trim Big Values
	const int tid = threadIdx1D();
	const int blockSize = blockSize1D();
	for (int i = tid; i < DESCRIPTOR_SIZE; i += blockSize)
		descriptors[i] = ::min(descriptors[i], MAGNITUDE_TH);
	__syncthreads();

	normalizeDescriptors(descriptors);
	__syncthreads();

	// Optional Step 9: Scale the result, so that it can be easily converted to byte array
	for (int k = tid; k < DESCRIPTOR_SIZE; k += blockSize)
		descriptors[k] = clip(INT_DESCR_FACTOR * descriptors[k]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void computePatchSIFTKernel(const PtrStepSzb image,
	const PtrStepSz<KeyPoint> keypoints, PtrStepf responses,
	float croppingScale, float keypointScale, double sigma)
{
	__shared__ uchar patch[PATCH_H][PATCH_W];
	__shared__ float histbuf[R_BINS + 2][C_BINS + 2][ORI_BINS + 2];
	__shared__ float descriptors[DESCRIPTOR_SIZE + 1];

	const int ix = blockIdx.x;
	if (ix >= keypoints.rows)
		return;

	// GaussianBlur(patch, img, Size(), sigma, sigma);
	rectifyPatch(image, keypoints[ix], patch, croppingScale);

	Histogram hist(histbuf, HistBin(PATCH_H, PATCH_W, keypointScale));

	if (threadIdx.x == 0 && threadIdx.y == 0)
		descriptors[0] = 1.f;
	hist.clear();
	__syncthreads();

	createHistogram(patch, hist, keypointScale);
	__syncthreads();

	describeFeatureVector(hist, &descriptors[1]);
	__syncthreads();

	const int tid = threadIdx1D();
	const int blockSize = blockSize1D();
	for (int k = tid; k < DESCRIPTOR_SIZE + 1; k += blockSize)
		responses(ix, k) = descriptors[k];
}

__global__ void binarizeDescriptorsKernel(const PtrStepf src, PtrStepSzb dst)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= dst.cols || iy >= dst.rows)
		return;

	uchar byte = 0;
	const float* ptrSrc = src.ptr(iy) + ix * 8;

	byte |= (*ptrSrc++ > 0) << 7;
	byte |= (*ptrSrc++ > 0) << 6;
	byte |= (*ptrSrc++ > 0) << 5;
	byte |= (*ptrSrc++ > 0) << 4;
	byte |= (*ptrSrc++ > 0) << 3;
	byte |= (*ptrSrc++ > 0) << 2;
	byte |= (*ptrSrc++ > 0) << 1;
	byte |= (*ptrSrc++ > 0) << 0;

	dst(iy, ix) = byte;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void computePatchSIFTs(const GpuMat& image, const GpuMat& keypoints, GpuMat& responses,
	float croppingScale, float keypointScale, double sigma, cudaStream_t stream)
{
	const dim3 dimBlock(SIFT_BLOCK_SIZE_X, SIFT_BLOCK_SIZE_Y);
	const dim3 dimGrid(keypoints.size().height, 1);

	sigma = sqrt(max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01));

	computePatchSIFTKernel<<<dimGrid, dimBlock, 0, stream>>>(image, keypoints, responses, croppingScale, keypointScale, sigma);

	CUDA_CHECK(cudaGetLastError());
}

void binarizeDescriptors(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
{
	constexpr int BLOCK_SIZE = 32;
	const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 dimGrid(divUp(dst.cols, BLOCK_SIZE), divUp(dst.rows, BLOCK_SIZE));

	binarizeDescriptorsKernel<<<dimGrid, dimBlock, 0, stream>>>(src, dst);

	CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace cuda
} // namespace cv
