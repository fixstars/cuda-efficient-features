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

void computePatchSIFTs(const GpuMat& image, const GpuMat& keypoints, GpuMat& responses,
	float croppingScale, float keypointScale = 1.f / 6, double sigma = 1.6, cudaStream_t stream = 0);

void binarizeDescriptors(const GpuMat& src, GpuMat& dst, cudaStream_t stream = 0);

} // namespace gpu
} // namespace cv
} // namespace cv
