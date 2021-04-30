"""
    计算alphapose加速比的方法
"""
import torch
import time
import os
import argparse
import numpy as np
from tools.trt_lite import TrtLite
from detector.yolo.darknet_trt import Darknet


def get_parser():
    parser = argparse.ArgumentParser(description='YOLOv3_SPP Demo')
    parser.add_argument('--cfg', type=str, default='./detector/yolo/cfg/yolov3-spp.cfg',
                        help='experiment configure file name')
    parser.add_argument('--weight', type=str, default='./detector/yolo/data/yolov3-spp.weights',
                        help='checkpoint file name')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--height', type=int, default=608, help='image height of input')
    parser.add_argument('--width', type=int, default=608, help='image width of input')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu id')
    parser.add_argument('--engine_path', type=str, default='./yolov3_spp_-1_608_608_dynamic_folded.engine',
                        help='the path of txt engine')
    args = parser.parse_args()
    return args


def run_yolov3(args):
    """
        运行alphapose模型，计算它的推理时间
    """
    # cfg = update_config(args.cfg)
    # 创建模型
    model = Darknet(args.cfg, args)
    model.net_info['height'] = 608
    # 加载权重
    print('Loading pose model from %s...' % (args.weight,))
    model.load_weights(args.weight)
    model = model.to(args.device)
    model = model.to('cuda:0')
    input_data = torch.randn(args.batch, 3, args.height, args.height, dtype=torch.float32).to('cuda:0')
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
    print('PyTorch time:', time_pytorch)
    return time_pytorch, output_data_pytorch


def run_trt(args):
    # 生成了两个trt模型
    engine_file_path = args.engine_path
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)

    print('====', engine_file_path, '===')
    trt = TrtLite(dll_file='./build/ScatterND.so', engine_file_path=engine_file_path)
    trt.print_info()
    # 这个形状可以不使用
    i2shape = {0: (args.batch, 3, args.height, args.height)}
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
    time_pytorch, output_data_pytorch = run_yolov3(arg)
    time_trt, output_data_trt = run_trt(arg)
    print('Speedup:', time_pytorch / time_trt)
    # print('Average diff percentage:',
    #       np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch)))
