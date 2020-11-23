## 注意事项
- 把/usr/local/cuda-10.0/include 拷贝到 /data/lupu/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/include/third_party/gpus/cuda/include 下，
该目录确实没有源码 https://github.com/tensorflow/tensorflow/issues/34428
- 由于C11 里面一些遗留问题，在Cmake中需要使用 d_glibcxx_use_cxx11_abi=0确保编译成功
- GCC 版本 7 (GCC 5.4 是不行的)
- Office reference [custome-op](https://github.com/tensorflow/custom-op)
- GCC 7编译出的版本session 执行segmentation 错误 (错误为layers\dcn\deformable_conv2d.cpp 中InferenceContext *c 为空)

## Session 可以使用的版本
- gcc 使用 4.8 [TF-编译对应版本](https://www.tensorflow.org/install/source#linux) [cmake 指定版本](https://stackoverflow.com/questions/17275348/how-to-specify-new-gcc-path-for-cmake)
- set(CMAKE_CUDA_FLAGS "-std=c++11 --expt-relaxed-constexpr") 添加-std=c++11, NVCC的编译也需使用c++11
- set(CMAKE_BUILD_TYPE RelWithDebInfo) 配合gdb python使用，对*.so的报错进行定位和调试
- target_link_libraries 不正确话会提示 cmake_link.o失败
- 参考 [](http://3ms.huawei.com/km/blogs/details/7872031) [](https://github.com/tensorflow/tensorflow/issues/30494)