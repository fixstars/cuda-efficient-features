# cuda-efficient-features

A CUDA implementation of keypoint detection and descriptor extraction

---

## Description
This project implements keypoint detection and descriptor extraction in CUDA,  
 motivated by [iago-suarez/efficient-descriptors](https://github.com/iago-suarez/efficient-descriptors) and provides the following features.

### Keypoint Detection
- Provides CUDA implementation of multi-scale FAST corner detection based on OpenCV's `cv::ORB::detect`
- Added functionality for controlling distribution of keypoints

The figure below shows a comparison of keypoint detection results between OpenCV's implementation and this project's one.
<div align="center">
<img src="https://github.com/fixstars/cuda-efficient-features/wiki/images/keypoints1.png" width=384> <img src="https://github.com/fixstars/cuda-efficient-features/wiki/images/keypoints2.png" width=384><br/>
</div>

**Left**: Even though OpenCV's implementation detected about 20,000 keypoints, most of them were concentrated in the leaves of trees.
**Right**: With this project's `nonmaxRadius` setting of 15, the keypoints are detected evenly throughout the image.

### Descriptor Extraction
- Provides CUDA implementation of **BAD** and **HashSIFT** descriptors proposed by Suarez et al. [1][2]

### OpenCV API
- Provides APIs in the same format as OpenCV's `cv::Feature2D`
	- `detect,compute,detectAndCompute`
- Also provides asynchronous APIs for concurrent execution
	- `detectAsync,computeAsync,detectAndComputeAsync`

---

## References
- [1] Suarez, Iago, Jose M. Buenaposada, and Luis Baumela. "Revisiting binary local image description for resource limited devices." IEEE Robotics and Automation Letters 6.4 (2021): 8317-8324.
- [2] https://github.com/iago-suarez/efficient-descriptors

---

## Performance
Using `sample_benchmark`, We measured the processing time for each API below.

- `detect`: keypoint detection only
- `compute`: descriptor extraction only
- `detectAndCompute`: keypoint detection and descriptor extraction

Each processing time below is an average of 11 images obtained from [SceauxCastle](https://github.com/openMVG/ImageDataset_SceauxCastle). The unit of time is milliseconds.

### detect
With the default parameters, we measured the processing time of keypoint detection while changing the image size to FHD(1920x1080), 4K(3840x2160), and 8K(7680x4320).

| Device        | FHD | 4K   | 8K   |
|---------------|-----|------|------|
| RTX 3060 Ti   | 1.6 | 2.9  | 5.5  |
| Jetson Xavier | 5.6 | 12.1 | 27.5 |

### compute
For each descriptor, we measured the processing time of descriptor extraction for 40,000 keypoints.

| Device        | BAD256 | BAD512 | HashSIFT256 | HashSIFT512 |
|---------------|--------|--------|-------------|-------------|
| RTX 3060 Ti   | 1.5    | 2.7    | 3.5         | 3.9         |
| Jetson Xavier | 19.1   | 28.2   | 21.9        | 24.8        |

### detectAndCompute
For each descriptor, we measured the processing time when executing both keypoint detection and descriptor extraction for 40,000 keypoints.

| Device        | BAD256 | BAD512 | HashSIFT256 | HashSIFT512 |
|---------------|--------|--------|-------------|-------------|
| RTX 3060 Ti   | 7.2    | 8.2    | 8.5         | 8.9         |
| Jetson Xavier | 41.7   | 48.8   | 46.2        | 49.2        |

---

## Requirements
|Package Name|Minimum Requirements|Note|
|---|---|---|
|CMake|version >= 3.18||
|CUDA Toolkit|compute capability >= 6.0|
|OpenCV|version >= 4.6.0||
|OpenCV CUDA module|version >= 4.6.0|included in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)|

---

## How to build
```
$ git clone https://github.com/fixstars/cuda-efficient-features.git
$ cd cuda-efficient-features
$ git submodule update --init  # needed if BUILD_TESTS is ON
$ mkdir build
$ cd build
$ cmake ../  # Several options available (e.g. -DBUILD_TESTS=ON -DCUDA_ARCHS=86)
$ make
```

### CMake options
|Option|Description|Default|
|---|---|---|
|BUILD_SAMPLES|Build samples|`ON`|
|BUILD_TESTS|Build tests|`OFF`|
|CUDA_ARCHS|List of architectures to generate device code for|`52;61;72;75;86`|

## How to run
### `samples`

|Command|Description|
|---|---|
|`./samples/sample_feature_extraction input-image [options]`|Feature detection and description|
|`./samples/sample_feature_matching first-image second-image [options]`|Feature matching on an image pair|
|`./samples/sample_image_sequence image-format [options]`|Feature matching on an image sequence|
|`./samples/sample_benchmark input-image [options]`|Performance benchmarking|
|`./samples/hpatches_description hpatchs-dir [options]`|Feature description on HPatches dataset<br>for [hpatches-benchmark](https://github.com/hpatches/hpatches-benchmark)|

Use the `--help` or `-h` option for detailed information.
```
./samples/sample_feature_extraction -h
```

### `tests`
Run the following command.
```
./tests/tests
```

---

## Author
The "adaskit Team"  

The adaskit is an open-source project created by [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

---

## License
Apache License 2.0
