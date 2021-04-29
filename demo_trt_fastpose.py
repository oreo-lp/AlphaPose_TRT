"""
    计算alphapose加速比的方法
"""
import torch
import torchvision
from torchsummary import summary
import time
import pycuda.driver as cuda
import pycuda.autoinit
from alphapose.models import builder
from easydict import EasyDict as edict
import yaml
import os
import argparse
import numpy as np
from tools.trt_lite import TrtLite


def get_parser():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str, default='./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str, default='./pretrained_models/fast_res50_256x192.pth',
                        help='checkpoint file name')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--height', type=int, default=256, help='image height of input')
    parser.add_argument('--width', type=int, default=192, help='image width of input')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    parser.add_argument('--engine_path', type=str, default='./alphaPose_-1_3_256_192_dynamic.engine',
                        help='the path of txt engine')
    args = parser.parse_args()
    return args


class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor

    def get_pointer(self):
        return self.tensor.data_ptr()


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def run_fastpose(args):
    """
        运行alphapose模型，计算它的推理时间
    """
    cfg = update_config(args.cfg)
    # 创建模型
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    # 加载权重
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model = pose_model.to('cuda:0')
    input_data = torch.randn(arg.batch, 3, 256, 192, dtype=torch.float32).to('cuda:0')
    # 转成numpy，用于对比加速结果
    output_data_pytorch = pose_model(input_data).cpu().detach().numpy()
    # 让模型跑100次，然后计算时间
    nRound = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        pose_model(input_data)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
    print('PyTorch time:', time_pytorch)
    return time_pytorch, output_data_pytorch


def run_trt(args):
    # 生成了两个trt模型
    engine_file_path = args.engine_path
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)

    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    # 这个形状可以不使用
    i2shape = {0: (args.batch, 3, 256, 192)}
    io_info = trt.get_io_info(i2shape)
    # 分配显存
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    # 保存输出的结果
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)
    input_data = torch.randn(args.batch, 3, args.height, args.height, dtype=torch.float32, device='cuda')
    d_buffers[0] = input_data
    trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    output_data_trt = d_buffers[1].cpu().numpy()
    nRound = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    torch.cuda.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)
    return time_trt, output_data_trt


if __name__ == '__main__':
    arg = get_parser()
    time_pytorch, output_data_pytorch = run_fastpose(arg)
    time_trt, output_data_trt = run_trt(arg)
    print('Speedup:', time_pytorch / time_trt)
    # print('Average diff percentage:',
    #       np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch)))
