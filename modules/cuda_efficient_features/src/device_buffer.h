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

#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <opencv2/core.hpp>

namespace cv
{
namespace cuda
{

class DeviceBuffer
{
public:

	// non-copiable
	//DeviceBuffer(const DeviceBuffer&) = delete;
	//DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	DeviceBuffer();
	~DeviceBuffer();

	void* allocate(size_t size);
	void* allocate(int rows, int cols, int type);
	GpuMat createMat(int rows, int cols, int type);

	void release();

private:

	void* data;
	size_t capacity;
};

} // namespace cuda
} // namespace cv

#endif // !__DEVICE_BUFFER_H__
