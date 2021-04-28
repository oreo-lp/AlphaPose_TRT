import argparse
import torch
from alphapose.models import builder
import yaml
from easydict import EasyDict as edict


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def get_parser():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str, default='./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str, default='./pretrained_models/fast_res50_256x192.pth',
                        help='checkpoint file name')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size')
    parser.add_argument('--height', type=int, default=256, help='image height of input')
    parser.add_argument('--width', type=int, default=192, help='image width of input')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    args = parser.parse_args()
    return args


def transform2onnx(args):
    cfg = update_config(args.cfg)
    # 创建模型
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    # 加载权重
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model = pose_model.to('cuda:0')
    input_names = ['input']
    output_names = ['output']
    # 判断batch size时候是变化的
    dynamic = False
    if args.batch_size <= 0:
        dynamic = True
    # batch_size是动态变化的
    if dynamic:
        # 创建虚拟的输入张量
        dummy_input = torch.randn(1, 3, args.height, args.width, dtype=torch.float32).to('cuda:0')
        onnx_file_name = "alphaPose_-1_3_{}_{}_dynamic.onnx".format(args.height, args.width)
        # dynamic_axes = {"input": [0, 2, 3]}
        dynamic_axes = {"input": {0: "batch_size", 2: "height", 3: "width"},
                        "output": {0: "batch_size", 2: "height", 3: "width"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(pose_model,
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
        # 创建虚拟的输入张量
        dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, dtype=torch.float32).to('cuda:0')
        onnx_file_name = "alphaPose_{}_3_{}_{}_dynamic.onnx".format(args.batch_size, args.height, args.width)
        print('Export the onnx model ...')
        # 将pytorch模型转成onnx模型
        torch.onnx.export(pose_model, dummy_input, onnx_file_name, input_names=input_names, output_names=output_names,
                          verbose=True, opset_version=11)


if __name__ == '__main__':
    args = get_parser()
    transform2onnx(args)
# onnx -> engine
# trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
