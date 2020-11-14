## 注意事项
- 把/usr/local/cuda-10.0/include 拷贝到 /data/lupu/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/include/third_party/gpus/cuda/include 下，
该目录确实没有源码 https://github.com/tensorflow/tensorflow/issues/34428
- 由于C11 里面一些遗留问题，在Cmake中需要使用 d_glibcxx_use_cxx11_abi=0确保编译成功
- GCC 版本 7 (GCC 5.4 是不行的)
- Office reference [custome-op](https://github.com/tensorflow/custom-op)