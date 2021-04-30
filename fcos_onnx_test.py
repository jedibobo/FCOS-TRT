import torch
import torchvision
# from torchsummary import summary
import time
# import pycuda.driver as cuda
# import pycuda.autoinit
from fcos_core.modeling.detector import build_detection_model
import argparse
from fcos_core.config import cfg
import argparse
import cv2, os

from fcos_core.config import cfg
from demo.predictor import COCODemo

import time


parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument(
    "--config-file",
    default="configs/fcos/fcos_imprv_X_101_64x4d_FPN_2x.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--weights",
    default="FCOS_imprv_X_101_64x4d_FPN_2x.pth",
    metavar="FILE",
    help="path to the trained model",
)
parser.add_argument(
    "--images-dir",
    default="demo/images",
    metavar="DIR",
    help="path to demo images directory",
)
parser.add_argument(
    "--min-image-size",
    type=int,
    default=800,
    help="Smallest size of the image to feed to the model. "
        "Model was trained with 800, which gives best results",
)
parser.add_argument(
    "opts",
    help="Modify model config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

# load config from file and command-line arguments
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.MODEL.WEIGHT = args.weights

cfg.freeze()

# The following per-class thresholds are computed by maximizing
# per-class f-measure in their precision-recall curve.
# Please see compute_thresholds_for_classes() in coco_eval.py for details.
thresholds_for_classes = [
    0.4923645853996277, 0.4928510785102844, 0.5040897727012634,
    0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
    0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
    0.43564629554748535, 0.6089804172515869, 0.666087806224823,
    0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
    0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
    0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
    0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
    0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
    0.523737907409668, 0.47027698159217834, 0.5103300213813782,
    0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
    0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
    0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
    0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
    0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
    0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
    0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
    0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
    0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
    0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
    0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
    0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
    0.5164387822151184, 0.540651798248291, 0.5323763489723206,
    0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
    0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
    0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
    0.4224434792995453, 0.4998510777950287
]

demo_im_names = os.listdir(args.images_dir)

# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(
    cfg,
    confidence_thresholds_for_classes=thresholds_for_classes,
    min_image_size=args.min_image_size
)

count=0
start=time.time()
for im_name in demo_im_names:
    img = cv2.imread(os.path.join(args.images_dir, im_name))
    count+=1
    if img is None:
        continue
    start_time = time.time()
    composite = coco_demo.run_on_opencv_image(img)
    # print(composite)
time_pytorch=(time.time()-start)/count
print("Pytorch average inference time: {:.2f}s".format(time_pytorch))

# input_data = torch.randn(1, 3, 800, 800, dtype=torch.float32, device='cuda')


# print(coco_demo.model(input_data)[0])
# output_data_pytorch = coco_demo.model(input_data).cpu().detach().numpy()

# input_names = ['input']
# output_names = ['output']
# torch.onnx.export(resnet50, input_data, 'resnet50.onnx', input_names=input_names, output_names=output_names, verbose=False, opset_version=11)
# torch.onnx.export(resnet50, input_data, 'resnet50.dynamic_shape.onnx', dynamic_axes={"input": [0, 2, 3]}, input_names=input_names, output_names=output_names, verbose=False, opset_version=11)

# #继续运行python代码前，先运行如下命令
# #trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
# #trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50_fp16.trt --fp16
# #以下命令不必运行，仅供参考
# #trtexec --verbose --onnx=resnet50.dynamic_shape.onnx --saveEngine=resnet50.dynamic_shape.trt --optShapes=input:1x3x1080x1920 --minShapes=input:1x3x1080x1920 --maxShapes=input:1x3x1080x1920

input_data = torch.randn(1, 3, 800, 1216, dtype=torch.float16, device='cuda')
nRound=10
from trt_lite import TrtLite
import numpy as np
import os



for engine_file_path in ['fcos101.trt', 'fcos101_fp16.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 3, 800, 1216)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)

    d_buffers[0] = input_data
    trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    output_data_trt = d_buffers[1].cpu().numpy()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    torch.cuda.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)

    print('Speedup:', time_pytorch / time_trt)

    # print('Average diff percentage:', np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch)))


print("========================dynamic shape inference test===================================")
shape_list=[{0: (1, 3, 800, 1216)},{0: (1, 3, 400, 800)}]
for engine_file_path in ['fcos101_dy.trt', 'fcos101_dy_fp16.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 3, 800, 1216)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)

    d_buffers[0] = input_data
    trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    output_data_trt = d_buffers[1].cpu().numpy()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    torch.cuda.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)

    print('Speedup:', time_pytorch / time_trt)
