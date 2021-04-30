"""
    计算alphapose(yolov3+fastpose)加速比的方法
"""
import torch
import time
from alphapose.models import builder
from easydict import EasyDict as edict
import yaml
import os
import argparse
import numpy as np
from tools.trt_lite import TrtLite
from detector.yolo.darknet_trt import Darknet


def get_parser():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--fastpose_cfg', type=str, default='./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='FastPose configure file name')
    parser.add_argument('--yolo_cfg', type=str, default='./detector/yolo/cfg/yolov3-spp.cfg',
                        help='YOLOv3_SPP configure file name')
    parser.add_argument('--weight', type=str, default='./detector/yolo/data/yolov3-spp.weights',
                        help='checkpoint file name')
    parser.add_argument('--checkpoint', type=str, default='./pretrained_models/fast_res50_256x192.pth',
                        help='checkpoint file name')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    parser.add_argument('--fastpose_engine', type=str, default='./alphaPose_-1_3_256_192_dynamic.engine',
                        help='the path of txt engine')
    parser.add_argument('--yolo_engine', type=str, default='./yolov3_spp_-1_608_608_dynamic_folded.engine',
                        help='the path of txt engine')
    args = parser.parse_args()
    return args


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def run_fastpose(args):
    """
        运行alphapose模型，计算它的推理时间
    """
    cfg = update_config(args.fastpose_cfg)
    # 创建fastpose的模型
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
    # print('PyTorch time:', time_pytorch)
    return time_pytorch, output_data_pytorch


def run_fastpose_trt(args):
    # 生成了两个trt模型
    engine_file_path = args.fastpose_engine
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
    input_data = torch.randn(args.batch, 3, 256, 192, dtype=torch.float32, device='cuda')
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
    # print('TensorRT time:', time_trt)
    return time_trt, output_data_trt


def run_yolov3(args):
    """
        运行alphapose模型，计算它的推理时间
    """
    # cfg = update_config(args.cfg)
    # 创建模型
    model = Darknet(args.yolo_cfg, args)
    model.net_info['height'] = 608
    # 加载权重
    print('Loading pose model from %s...' % (args.weight,))
    model.load_weights(args.weight)
    model = model.to(args.device)
    model = model.to('cuda:0')
    input_data = torch.randn(args.batch, 3, 608, 608, dtype=torch.float32).to('cuda:0')
    # 转成numpy，用于对比加速结果
    output_data_pytorch = model(input_data).cpu().detach().numpy()
    # 让模型跑10次，然后计算时间
    nRound = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        model(input_data)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
    # print('PyTorch time:', time_pytorch)
    return time_pytorch, output_data_pytorch


def run_yolov3_trt(args):
    # 生成了两个trt模型
    engine_file_path = args.yolo_engine
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)

    print('====', engine_file_path, '===')
    trt = TrtLite(dll_file='./build/ScatterND.so', engine_file_path=engine_file_path)
    trt.print_info()
    # 这个形状可以不使用
    i2shape = {0: (args.batch, 3, 608, 608)}
    io_info = trt.get_io_info(i2shape)
    # 分配显存
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    # 保存输出的结果
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)
    input_data = torch.randn(args.batch, 3, 608, 608, dtype=torch.float32, device='cuda')
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
    # print('TensorRT time:', time_trt)
    return time_trt, output_data_trt


def run_pytorch(args):
    # 执行fastpose + yolo
    time_fastpose, _ = run_fastpose(args)
    time_yolov3, _ = run_yolov3(args)
    return time_fastpose + time_yolov3


def run_trt(args):
    time_fastpose_trt, _ = run_fastpose_trt(args)
    time_yolov3_trt, _ = run_yolov3_trt(args)
    return time_fastpose_trt + time_yolov3_trt


if __name__ == '__main__':
    arg = get_parser()
    time_pytorch = run_pytorch(arg)
    print("Pytorch time:", time_pytorch)
    time_trt = run_trt(arg)
    print('TensorRT time:', time_trt)
    print('Speedup:', time_pytorch / time_trt)
    # print('Average diff percentage:',
    #       np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch)))
