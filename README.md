# FCOS-TRT #TRT2021

安装步骤参考FCOS的Github，Install.md。[FCOS/INSTALL.md at master · tianzhi0549/FCOS (github.com)](https://github.com/tianzhi0549/FCOS/blob/master/INSTALL.md) 注意pytorch的版本一定要是1.4.0的。

benchmark结果如下：

使用的模型是FCOS_imprv_X_101_64x4d_FPN_2x.pth以及导出的trt文件。

| 框架/时间              | 输入800*1216 | 加速比 |
| ---------------------- | ------------ | ------ |
| Pytorch                | 320ms        | 1      |
| TRT(static shape)      | 60.5ms       | 5.347  |
| TRT-fp16(static shape) | 72.4ms       | 4.471  |

由于dynamic shape存在问题，没有方法测试。导出的中间格式onnx格式为dynamic shape的，[1,3,-1,-1].



测试方法：

进入名字为fcos的docker,进入相应的环境:

```
docker start fcos 
docker attach fcos
conda activate fcos38
```

执行fcos_onnx_test.py文件

```
python fcos_onnx_test.py
```



至于中间出现的问题：

```
[TensorRT] ERROR: ../rtSafe/cuda/cudaConvolutionRunner.cpp (483) - Cudnn Error in executeConv: 3 (CUDNN_STATUS_BAD_PARAM)
```

我通过查看所有系统的cudnn环境，并没有发现相关的7.6.3的，所以推测这是该torch环境编译时保存的，（包括其保存的cuda10.1）。可以通过如下测试证明实际调用的cuda库：

进入python环境：

```
>>> import torch
>>> import torch.utils
>>> import torch.utils.cpp_extension
>>> torch.utils.cpp_extension.CUDA_HOME     
```

