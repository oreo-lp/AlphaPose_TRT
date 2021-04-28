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
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='dynamic shape')
    parser.add_argument('--height', type=int, default=608, help='image height of input')
    parser.add_argument('--width', type=int, default=608, help='image width of input')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    args = parser.parse_args()
    return args


# batch size == 1
def transform_to_onnx(cfgfile, args):
    model = Darknet(cfgfile, args)
    model.net_info['height'] = 608
    model.load_weights(args.weight)
    model = model.to(args.device)
    input_names = ["input"]
    output_names = ["output"]
    dynamic = False
    if args.batch_size <= 0:
        dynamic = True
    if dynamic:
        # 创建虚拟的输入张量
        dummy_input = torch.randn(1, 3, args.height, args.width, dtype=torch.float32).to('cuda:0')
        onnx_file_name = "yolov3_spp_{}_{}_{}dynamic.onnx".format(args.batch_size, args.height, args.width)
        # dynamic_axes = {"input": [0, 2, 3]}
        dynamic_axes = {"input": {0: "batch_size"},
                        "output": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name
    else:
        dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, dtype=torch.float32, device='cuda')
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
    transform_to_onnx(args.cfg, args)

# onnx -> engine
# trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
# /opt/tensorrt/bin/trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
