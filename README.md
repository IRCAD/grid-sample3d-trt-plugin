## TensorRT plugin for 3D GridSample Operator

At current stage, TensorRT(up to version 8.6.1) does not support 3D GridSample operator.

This plugin is a custom implementation of the 3D GridSample operator for TensorRT. It is inspired by the [GridSample operator](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) from PyTorch, and the code structure is inspired by project [onnxparser-trt-plugin-sample](https://github.com/TrojanXu/onnxparser-trt-plugin-sample).

### Installation

1. Let cmake find nvcc
```shell
export PATH=/usr/local/cuda/bin:$PATH
```

2. Locate TensorRT on your system. TensorRT is not bundled with the CUDA
   toolkit, so its headers and libraries need to be pointed to explicitly:
   - `-DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include` — directory
     containing `NvInfer.h` (e.g. a downloaded TensorRT SDK's `include/`
     directory, or `/usr/include/x86_64-linux-gnu` if TensorRT was installed
     via apt).
   - `-DTENSORRT_LIB_DIR=/path/to/dir` — directory containing
     `libnvinfer.so[.N]` (e.g. a conda/venv's
     `lib/pythonX.Y/site-packages/tensorrt_libs` if TensorRT was installed
     via the `tensorrt` pip package, or `/usr/lib/x86_64-linux-gnu` if
     installed via apt). If omitted, the build auto-detects this from the
     active Python environment's `tensorrt_libs` package (useful when
     building inside a conda env that already has `tensorrt` installed via
     pip so the plugin links against the same TensorRT version used at
     runtime).

3. Build the plugin with the following commands:
```shell
cmake -B build -S . \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DTENSORRT_LIB_DIR=/path/to/dir/containing/libnvinfer.so
cmake --build build -j$(nproc)
```

   This builds `libgrid_sample_3d_plugin.so` plus a standalone C++ test
   executable (`build/test/test_grid_sample`, run with `cd build && ctest`).
   If only the shared library is needed (e.g. for a Python-only consumer),
   skip the test executable with `-DBUILD_TESTS=OFF`.

### Usage 

for python code (only on Linux platform), load the plugin with:

```python
import ctypes
success = ctypes.CDLL("build/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)
```

see [test_grid_sample3d.py](./test/test_grid_sample3d_plugin.py) for more details.

## Tensor precision

The plugin supports both TensorRT `FLOAT` and `HALF` formats. The image input,
grid input, and output must use the same format. A newly created plugin defaults
to `HALF`, preserving the low-memory inference behavior used by existing
consumers; TensorRT selects `FLOAT` when the network tensors are explicitly
`FLOAT`. Execution dispatch uses the datatype from the current TensorRT
execution descriptors, so a dynamic-shape profile cannot leave `enqueue` using
a stale datatype from an earlier configuration.

The TensorRT integration test checks both supported precisions, output type
propagation, and rejection of mixed image/grid formats.
