"""
    该脚本是将darknet转成onnx模型
"""
from detector.yolo.darknet import *
import torch
from detector.yolo.darknet import Darknet
from easydict import EasyDict as edict
import yaml
import argparse


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def get_parser():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str, default='./detector/yolo/cfg/yolov3-spp.cfg',
                        help='experiment configure file name')
    parser.add_argument('--weight', type=str, default='./detector/yolo/data/yolov3-spp.weights',
                        help='checkpoint file name')
    parser.add_argument('--height', type=int, default=608, help='image height of input')
    parser.add_argument('--width', type=int, default=608, help='image width of input')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    args = parser.parse_args()
    return args


# batch size == 1
def transform_to_onnx(cfgfile, args, batch_size=1):
    model = Darknet(cfgfile, args)
    model.net_info['height'] = 608
    model.load_weights(args.weight)
    model = model.to(args.device)
    input_names = ["input"]
    output_names = ["boxes"]
    dummy_input = torch.randn(1, 3, 608, 608, dtype=torch.float32, device='cuda')
    # 输出的尺寸: torch.Size([1, 22743, 85])
    # print(model(dummy_input).shape)
    onnx_name = 'yolov3_static.onnx'
    torch.onnx.export(model,
                      dummy_input,
                      onnx_name,
                      opset_version=11,
                      verbose=True,
                      do_constant_folding=True,
                      dynamic_axes=None,
                      input_names=input_names, output_names=output_names)
    print('Onnx model exporting done')


if __name__ == '__main__':
    args = get_parser()
    batch_size = 1
    transform_to_onnx(args.cfg, args, batch_size)

# onnx -> engine
# trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
# /opt/tensorrt/bin/trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
