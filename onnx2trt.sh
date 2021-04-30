# python onnx/export_model_to_onnx.py &&
# python export_fcos_dynamicshape.py && 
# trtexec --verbose --onnx=fcos101.onnx --saveEngine=fcos101.trt &&
# trtexec --verbose --onnx=fcos101.onnx --saveEngine=fcos101_fp16.trt --fp16 &&
trtexec --verbose --onnx=fcos101_dynamic.onnx --saveEngine=fcos101_dy.trt --optShapes=input:1x3x800x1216 --minShapes=input:1x3x400x800 --maxShapes=input:1x3x1600x2400 &&
trtexec --verbose --onnx=fcos101_dynamic.onnx --saveEngine=fcos101_dy_fp16.trt --fp16 --optShapes=input:1x3x800x1216 --minShapes=input:1x3x400x800 --maxShapes=input:1x3x1600x2400

# trtexec --verbose --onnx=fcos101_dynamic.onnx --saveEngine=fcos101_dy.trt --optShapes=input_image:1x3x800x1216 --minShapes=input_image:1x3x400x800 --maxShapes=input_image:1x3x1600x2400 &&
# trtexec --verbose --onnx=fcos101_dynamic.onnx --saveEngine=fcos101_dy_fp16.trt --fp16 --optShapes=input_image:1x3x800x1216 --minShapes=input_image:1x3x400x800 --maxShapes=input_image:1x3x1600x2400