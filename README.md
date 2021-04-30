# FCOS-TRT

安装步骤参考FCOS的Github，Install.md。[FCOS/INSTALL.md at master · tianzhi0549/FCOS (github.com)](https://github.com/tianzhi0549/FCOS/blob/master/INSTALL.md) 注意pytorch的版本一定要是1.4.0的。

benchmark结果如下：

使用的模型是FCOS_imprv_X_101_64x4d_FPN_2x.pth以及导出的trt文件。

| 框架/时间 | 输入800*1216 | 加速比 |
| --------- | ------------ | ------ |
| Pytorch   | 320ms        | 1      |
| TRT       | 60.5ms       | 5.347  |
| TRT-fp16  | 72.4ms       | 4.471  |

由于dynamic shape存在问题，没有方法测试。导出的中间格式onnx格式为dynamic shape的，[1,3,-1,-1].