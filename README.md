## Requirements
* CUDA 11.1
* TensorRT 7.2.2
* Python 3.8.5
* Cython
* PyTorch 1.8.1
* torchvision 0.9.1
* numpy 1.17.4 (numpy版本过高会出报错[this issue]: https://github.com/MVIG-SJTU/AlphaPose/issues/777)
* python-package setuptools >= 40.0, reported by [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/838)

## Results
AlphaPose在数据预处理部分使用YOLOv3-SPP模型检测出一幅图像中的多个人体，然后分别将这些人体送入到FastPose模型中进行姿态估计。
因此，我们对YOLOv3-SPP模型以及FastPose模型都进行了加速。下面的表格列举了加速前后的mAP值：
Environment Results on COCO val 2017 (Tesla T4):
<center>

| Method | ground truth box mAP@0.6 | detection boxes mAP@0.6 | 
|:-------|:-----:|:-------:|
| AlphaPose | 0.743 |0.718 | 
| **AlphaPose_trt** | **0.743** | **0.718** |

</center>


### YOLOv3-SPP speed up
下面的表格列举了YOLOv3-SPP模型的加速比信息：
<center>

| model | Batchsize | Latency (ms) | Throughput  | Latency Speedup |Throughput speedup|
|:-------|:-----:|:-------:|:-----:|:-------:|:-------:|
| AlphaPose | 1 | 0.053532 | 18680.42 | 2.5x | 2.5x |
|  | 2 | 0.092021 | 21734.17 | 2.7x | 2.7x |
|  | 4 | 0.168305 | 23766.38 | 2.8x | 2.8x |
|  | 8 | 0.316017 | 25315.09 | 2.8x | 2.8x |
| **AlphaPose_trt** | **1** | **0.021086** | ***47424.83*** | | |
|  | **2** | **0.034413** | ***58117.57*** | | |
|  | **4** | **0.060228** | ***66414.29*** | | |
|  | **8** | **0.113515** | ***70475.14*** | | |


</center>

### Pose Estimation speed up
下面的表格列举了YOLOv3-SPP模型的加速比信息：
<center>

| model | Batchsize | Latency (ms) | Throughput  | Latency Speedup |Throughput speedup|
|:-------|:-----:|:-------:|:-----:|:-------:|:-------:|
| AlphaPose | 1 | 0.023896 | 41848.00 | 9x | 9x |
|  | 2 | 0.024911 | 80285.82 | 7x | 7x |
|  | 4 | 0.027914 | 143297.27 | 5.8x | 5.8x |
|  | 8 | 0.033429 | 239313.17 | 4x | 4x |
|  | 16 | 0.056172 | 284839.42 | 3.6x | 3.6x |
|  | 32 | 0.104112 | 307361.30 | 3.5x | 3.5x |
|  | 64 | 0.202907 | 315415.43 | 3.4x | 3.4x |
| **AlphaPose_trt** | **1** | **0.002631** | ***380083.62*** | | |
|  | **2** | **0.003602** | ***555247.08*** | | |
|  | **4** | **0.004772** | ***838222.97*** | | |
|  | **8** | **0.008537** | ***937097.34*** | | |
|  | **16** | **0.015657** | ***1021907.13*** | | |
|  | **32** | **0.029754** | ***1075485.65*** | | |
|  | **64** | **0.058865** | ***1087233.50*** | | |

</center>

### Code installation
   AlphaPose的安装参考自：https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md
#### (Recommended) Install with conda

Install conda from [here](https://repo.anaconda.com/miniconda/), Miniconda3-latest-(OS)-(platform).
```shell
# 1. Create a conda virtual environment.
conda create -n alphapose python=3.6 -y
conda activate alphapose

# 2. Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0

# 3. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
# git pull origin pull/592/head if you use PyTorch>=1.5
cd AlphaPose


# 4. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python -m pip install cython
sudo apt-get install libyaml-dev
################Only For Ubuntu 18.04#################
locale-gen C.UTF-8
# if locale-gen not found
sudo apt-get install locales
export LANG=C.UTF-8
######################################################
python setup.py build develop
```

#### Install with pip
```shell
# 1. Install PyTorch
pip3 install torch==1.1.0 torchvision==0.3.0

# Check torch environment by:  python3 -m torch.utils.collect_env

# 2. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
# git pull origin pull/592/head if you use PyTorch>=1.5
cd AlphaPose

# 3. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
pip install cython
sudo apt-get install libyaml-dev
python3 setup.py build develop --user
```

## YOLOv3-SPP to engine
### 1. 生成static shape的engine模型
(1) YOLOv3-SPP转成onnx模型，输入数据的尺寸默认为: 1x3x608x608
``` shell
python ./darknet2onnx.py 
--cfg ./detector/yolo/cfg/yolov3-spp.cfg 
--weight ./detector/yolo/data/yolov3-spp.weights
```
执行该语句之后，会在当前目录下产生一个yolov3_spp_static.onnx模型

(2) 由于YOLOv3-SPP模型中存在Padding操作，trt不能直接识别，因此需要onnx进行修改
``` shell
polygraphy surgeon sanitize yolov3_spp_static.onnx 
--fold-constants 
--output yolov3_spp_static_folded.onnx
```
参考信息：https://github.com/onnx/onnx-tensorrt/blob/master/docs/faq.md#inputsat0-must-be-an-initializer-or-inputsat0is_weights

(3) 由onnx模型生成engine
需要注册ScatterND plugin, 参考地址：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/plugins
``` shell 
trtexec --onnx=yolov3_spp_static_folded.onnx 
--explicitBatch 
--saveEngine=yolov3_spp_static_folded.engine 
--workspace=10240 --fp16 --verbose 
--plugins=build/ScatterND.so
```

### 2. 生成dynamic shape的engine模型
(1)  YOLOv3-SPP转成onnx模型，输入数据的默认尺寸为: -1x3x608x608 (-1表示batch size可变)
``` shell
python darknet2onnx_dynamic.py 
--cfg ./detector/yolo/cfg/yolov3-spp.cfg 
--weight ./detector/yolo/data/yolov3-spp.weights
```
执行该语句之后，会在当前目录下产生一个yolov3_spp_-1_608_608_dynamic.onnx模型

(2) 对onnx模型就行修改
``` shell
polygraphy surgeon sanitize yolov3_spp_-1_608_608_dynamic.onnx 
--fold-constants 
--output yolov3_spp_-1_608_608_dynamic_folded.onnx
```

(3) 由onnx模型转成engine
``` shell
trtexec --onnx=yolov3_spp_-1_608_608_dynamic_folded.onnx 
--explicitBatch 
--saveEngine=yolov3_spp_-1_608_608_dynamic_folded.engine 
--workspace=10240 --fp16 --verbose 
--plugins=build/ScatterND.so 
--minShapes=input:1x3x608x608 
--optShapes=input:1x3x608x608 
--maxShapes=input:64x3x608x608 
--shapes=input:1x3x608x608
```

## FastPose to engine
### 1. 生成static shape的engine模型
(1) FastPose转成onnx模型. 模型输入数据的默认尺寸为: 1x3x256x192
``` shell
python pytorch2onnx.py --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 
--checkpoint ./pretrained_models/fast_res50_256x192.pth
```
执行完该指令之后，会在当前目录下生成一个fastPose.onnx模型

(2) onnx转成engine模型
```shell
trtexec trtexec --onnx=fastPose.onnx 
-saveEngine=fastPose.engine --workspace=10240 
--fp16 
--verbose
```
执行该命令之后，会在当前目录下生成一个fastPose.engine模型

### 2. 生成dynamic shape的engine模型
(1) 生成onnx模型，模型输入数据的默认尺寸为：-1x3x256x192 (-1表示batch size可变)
```shell
python pytorch2onnx_dynamic.py 
--cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 
--checkpoint ./pretrained_models/fast_res50_256x192.pth
```
执行该命令之后，会在当前目录下生成一个alphaPose_-1_3_256_192_dynamic.onnx模型

(2) onnx模型转成engine模型
```shell 
trtexec --onnx=alphaPose_-1_3_256_192_dynamic.onnx 
--saveEngine=alphaPose_-1_3_256_192_dynamic.engine 
--workspace=10240 --fp16 --verbose 
--minShapes=input:1x3x256x192 
--optShapes=input:1x3x256x192 
--maxShapes=input:128x3x256x192 
--shapes=input:1x3x256x192 
--explicitBatch
```
执行该命令之后，会在当前目录下生成一个alphaPose_-1_3_256_192_dynamic.engine模型


## inference
这一部分主要使用两个加速模型对图像以及视频进行检测
### 对图像进行检测
将图像放在example/demo文件夹下，然后执行下面的指令，检测结果将保存在examples/res/vis文件夹下
1. 使用未加速模型对图像进行检测
```shell
python inference.py --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 
--checkpoint ./pretrained_models/fast_res50_256x192.pth  
--save_img  --showbox 
--indir ./examples/demo
```
2. 使用tensorRT加速模型对图像进行检测
```shell
python trt_inference.py --yolo_engine ./yolov3_spp_static_folded.engine --pose_engine ./fastPose.engine --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint ./pretrained_models/fast_res50_256x192.pth --save_img  --indir ./examples/demo --dll_file ./build/ScatterND.so
```

### 对视频进行检测
将视频放在videmo文件夹下，推理的结果将保存在examples/res文件夹下
1. 使用未加速模型对图像进行检测
```shell
python inference.py --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
--checkpoint ./pretrained_models/fast_res50_256x192.pth 
--save_video
--video ./videos/demo.avi
```
2. 使用tensorRT加速模型对图像进行检测
```shell
python trt_inference.py --yolo_engine ./yolov3_spp_static_folded.engine
--cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
--checkpoint ./pretrained_models/fast_res50_256x192.pth 
--save_video
--video ./videos/demo_short.avi 
--dll_file ./build/ScatterND.so
--pose_engine ./fastPose.engine 
--detector yolo
```


## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @article{li2018crowdpose,
      title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
      author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
      journal={arXiv preprint arXiv:1812.00324},
      year={2018}
    }

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }

