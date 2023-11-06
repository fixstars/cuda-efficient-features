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

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

namespace cv
{
namespace cuda
{
namespace gpu
{

// Struct representing a pair of boxes in the patch
struct BoxPairParams
{
	int x1, x2, y1, y2, boxRadius;
};

void loadBoxPairParams(int paramSize);

void computeBAD(const GpuMat& integral, const GpuMat& keypoints, GpuMat& descriptors,
	float scaleFactor, int paramSize, Size patchSize, cudaStream_t stream);

void calcIntegralImage(const GpuMat& src, GpuMat& dst, Stream& stream);

} // namespace gpu
} // namespace cuda
} // namespace cv
