#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import reduce
import tensorrt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import tensorrt as trt
import cv2
import time
import ctypes


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor

    def get_pointer(self):
        return self.tensor.data_ptr()


class TrtTiny(object):
    def __init__(self, batch_size, out_height, out_width, engine_path, cuda_ctx=None, dll_file=None, mode="fastpose",
                 maxBs=16):
        super(TrtTiny, self).__init__()
        self.mode = mode
        self.dll_file = dll_file
        self.out_height = out_height
        self.out_width = out_width
        # register plugin
        self.register_plugin()
        self.cuda_ctx = cuda_ctx
        self.logger = trt.Logger()
        self.engine_path = engine_path
        self.engine = self._get_engine()  # create engine
        self.context = self.engine.create_execution_context()  # create context
        self.batch_size = batch_size
        self.maxBs = maxBs
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self._allocate_buffers()

    def register_plugin(self):
        # register plugin
        if self.dll_file is not None:
            ctypes.cdll.LoadLibrary(self.dll_file)
        self.logger = tensorrt.Logger(tensorrt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, '')

    def _get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def detect_context(self, img_in):
        # 设置当前数据的尺寸
        for binding in self.engine:
            dims = self.engine.get_binding_shape(binding)
            if dims[0] < 0 and binding == 'input':
                self.context.set_binding_shape(binding=0, shape=img_in.shape)
            # print("detect_context dims = {}".format(self.context.get_binding_shape(0 if binding == 'input' else 1)))
        # 计算推理时间
        img_in = np.ascontiguousarray(img_in)
        self.inputs[0].host = img_in
        if self.cuda_ctx:
            self.cuda_ctx.push()
        ta = time.time()
        trt_outputs = self._do_inference()
        tb = time.time()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        print('-----------------------------------')
        print('    {%s} TRT inference time: %f' % (self.engine_path, tb - ta))
        print('-----------------------------------')
        # pose -> (1,17,64,48), yolov3 -> (1,22743,85)
        if self.mode == 'fastpose':
            return trt_outputs[0].reshape(-1, 17, 64, 48)
        else:
            return trt_outputs[0].reshape(-1, self.out_height, self.out_width)

    def _allocate_buffers(self):
        for binding in self.engine:
            # print("binding  = {}".format(binding))  # input and output
            dims = self.engine.get_binding_shape(binding)
            # print("dims = {}".format(dims)) -> (1, 3, 256, 192); -> (1, 17, 64, 48)
            # print("before dims = {}".format(dims))
            # 使用context获取最大的显存，并进行分配
            if dims[0] < 0:
                # 要设置最大的尺寸，一次性给给运行环境分配最大的显存，后面到真实数据的时候再对context的输入进行改变
                if binding == 'input':
                    self.context.set_binding_shape(binding=0, shape=(self.maxBs, 3, dims[2], dims[3]))
                size = trt.volume(self.context.get_binding_shape(0 if binding == 'input' else 1))
            else:
                # 下面两种方法都可以
                # size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
                size = trt.volume(self.context.get_binding_shape(0 if binding == 'input' else 1)) * self.batch_size
            # print("after dims = {}".format(self.context.get_binding_shape(0 if binding == 'input' else 1)))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def _do_inference(self):
        # Transfer input data to the GPU.(optionally serialized via stream)
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.(optionally serialized via stream)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]


class TrtLite:
    def __init__(self, dll_file=None, build_engine_proc=None, build_engine_params=None, engine_file_path=None):

        logger = tensorrt.Logger(tensorrt.Logger.INFO)
        # add plugins
        if dll_file is not None:
            ctypes.cdll.LoadLibrary(dll_file)
        trt.init_libnvinfer_plugins(logger, '')
        if engine_file_path is None:
            with tensorrt.Builder(logger) as builder:
                if build_engine_params is not None:
                    self.engine = build_engine_proc(builder, *build_engine_params)
                else:
                    self.engine = build_engine_proc(builder)
        else:
            with open(engine_file_path, 'rb') as f, tensorrt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def __del__(self):
        self.engine = None
        self.context = None

    def save_to_file(self, engine_file_path):
        with open(engine_file_path, 'wb') as f:
            f.write(self.engine.serialize())

    def get_io_info(self, input_desc):
        def to_numpy_dtype(trt_dtype):
            tb = {
                tensorrt.DataType.BOOL: np.dtype('bool'),
                tensorrt.DataType.FLOAT: np.dtype('float32'),
                tensorrt.DataType.HALF: np.dtype('float16'),
                tensorrt.DataType.INT32: np.dtype('int32'),
                tensorrt.DataType.INT8: np.dtype('int8'),
            }
            return tb[trt_dtype]

        if isinstance(input_desc, dict):
            if self.engine.has_implicit_batch_dimension:
                print('Engine was built with static-shaped input so you should provide batch_size instead of i2shape')
                return
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            return [(self.engine.get_binding_name(i), self.engine.binding_is_input(i),
                     self.context.get_binding_shape(i), to_numpy_dtype(self.engine.get_binding_dtype(i))) for i in
                    range(self.engine.num_bindings)]

        batch_size = input_desc
        return [(self.engine.get_binding_name(i), self.engine.binding_is_input(i),
                 (batch_size,) + tuple(self.context.get_binding_shape(i)),
                 to_numpy_dtype(self.engine.get_binding_dtype(i))) for i in range(self.engine.num_bindings)]

    def allocate_io_buffers(self, input_desc, on_gpu):
        io_info = self.get_io_info(input_desc)
        if io_info is None:
            return
        if on_gpu:
            return [cuda.mem_alloc(reduce(lambda x, y: x * y, i[2]) * i[3].itemsize) for i in io_info]
        else:
            return [np.zeros(i[2], i[3]) for i in io_info]

    def execute(self, bindings, input_desc, stream_handle=0, input_consumed=None):
        if isinstance(input_desc, dict):
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            self.context.execute_async_v2(bindings, stream_handle, input_consumed)
            return

        batch_size = input_desc
        self.context.execute_async(batch_size, bindings, stream_handle, input_consumed)

    def print_info(self):
        print("Batch dimension is", "implicit" if self.engine.has_implicit_batch_dimension else "explicit")
        for i in range(self.engine.num_bindings):
            print("input" if self.engine.binding_is_input(i) else "output",
                  self.engine.get_binding_name(i), self.engine.get_binding_dtype(i),
                  self.engine.get_binding_shape(i),
                  -1 if -1 in self.engine.get_binding_shape(i) else reduce(
                      lambda x, y: x * y, self.engine.get_binding_shape(i)) * self.engine.get_binding_dtype(i).itemsize)
