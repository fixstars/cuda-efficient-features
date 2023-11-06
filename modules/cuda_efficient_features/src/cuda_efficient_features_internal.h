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

#ifndef __CUDA_EFFICIENT_FEATURES_INTERNAL_H__
#define __CUDA_EFFICIENT_FEATURES_INTERNAL_H__

#include <variant>
#include <opencv2/core/cuda.hpp>

namespace cv
{
namespace cuda
{

using KeyPoints = std::vector<KeyPoint>;
using InputKeyPoints = const std::variant<_InputArray, KeyPoints>&;

void getInputMat(InputArray src, GpuMat& dst, Stream& stream = Stream::Null());
void getOutputMat(OutputArray src, GpuMat& dst, int rows, int cols, int type);
void getKeypointsMat(InputKeyPoints src, GpuMat& dst, Stream& stream = Stream::Null());
bool isEmpty(InputKeyPoints keypoints);

} // namespace cuda
} // namespace cv

#endif // !__CUDA_EFFICIENT_FEATURES_INTERNAL_H__
