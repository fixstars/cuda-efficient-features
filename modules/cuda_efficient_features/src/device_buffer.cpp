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

#include "device_buffer.h"

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{

static constexpr int PITCH_ALIGN = 256;
static constexpr int ALIGN_MASK = PITCH_ALIGN - 1;
static constexpr int alignUp(int x) { return (x + ALIGN_MASK) & ~ALIGN_MASK; }

DeviceBuffer::DeviceBuffer() : data(nullptr), capacity(0)
{
}

DeviceBuffer::~DeviceBuffer()
{
	release();
}

void* DeviceBuffer::allocate(size_t size)
{
	if (size > capacity)
	{
		release();
		CUDA_CHECK(cudaMalloc(&data, size));
		capacity = size;
	}

	return data;
}

GpuMat DeviceBuffer::createMat(int rows, int cols, int type)
{
	if (rows == 1 || cols == 1)
	{
		const size_t size = rows * cols * getElemSize(type);
		allocate(size);
		return GpuMat(rows, cols, type, data);
	}
	else
	{
		const size_t step = alignUp(cols) * getElemSize(type);
		const size_t size = rows * step;
		allocate(size);
		return GpuMat(rows, cols, type, data, step);
	}
}

void DeviceBuffer::release()
{
	if (data)
		CUDA_CHECK(cudaFree(data));
	data = nullptr;
	capacity = 0;
}

} // namespace cuda
} // namespace cv
