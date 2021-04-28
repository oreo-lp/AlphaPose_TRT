import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tools.trt_lite import TrtTiny
import random


def run_alphaPose():
    random.seed(0)
    img_in = np.random.randn(2, 3, 256, 192).astype(np.float32)
    img_in_2 = np.random.randn(4, 3, 256, 192).astype(np.float32)
    # img_in = torch.randn(1, 3, 256, 192, dtype=torch.float32, device='cuda') # no torch
    # trt_model = TrtTiny(batch_size=1, out_height=22743, out_width=85, engine_path='./yolov3_spp_static_folded.engine',
    #                     cuda_ctx=pycuda.autoinit.context,
    #                     dll_file='./build/ScatterND.so', mode='yolo')
    trt_model = TrtTiny(batch_size=1, out_height=64, out_width=48,
                        engine_path='./alphaPose_-1_3_256_192_dynamic.engine',
                        cuda_ctx=pycuda.autoinit.context)
    # trt_model_copy = TrtTiny(batch_size=2, out_height=22743, out_width=85,
    #                          engine_path='./fastPose.engine',
    #                          cuda_ctx=pycuda.autoinit.context)
    for i in range(2):
        out = trt_model.detect_context(img_in)
        # out = trt_model.detect_context(img_in)
        print(out.shape)
        print(" ============================== ")
        print(out if i == 1 else out[:2, :, :, :])

    #
    # for i in range(2):
    #     out = trt_model_copy.detect_context(img_in)
    #     # out = trt_model.detect_context(img_in)
    #     print(out.shape)


def run_yolov3():
    random.seed(0)
    img_in = np.random.randn(2, 3, 608, 608).astype(np.float32)
    img_in_2 = np.random.randn(4, 3, 608, 608).astype(np.float32)
    trt_model = TrtTiny(batch_size=1, out_height=22743, out_width=85,
                        engine_path='./yolov3_spp_-1_608_608_dynamic_folded.engine',
                        cuda_ctx=pycuda.autoinit.context, mode='yolo',dll_file='./build/ScatterND.so')
    for i in range(2):
        out = trt_model.detect_context(img_in if i == 0 else img_in_2)
        # out = trt_model.detect_context(img_in)
        print(out.shape)


if __name__ == '__main__':
    run_yolov3()
