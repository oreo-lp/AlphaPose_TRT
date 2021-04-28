## 1. Requirements
* CUDA 11.1
* TensorRT 7.2.2
* Python 3.8.5
* Cython
* PyTorch 1.8.1
* torchvision 0.9.1
* numpy 1.17.4 (numpy版本过高会出报错 [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/777)
* python-package setuptools >= 40.0， reported by [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/838)

## 2. Results
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md) 存在多个目标检测+姿态估计模型的组合，
本项目(fork from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) )仅对[YOLOv3_SPP](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg) + [Fast Pose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)
进行加速。AlphaPose在数据预处理部分使用YOLOv3-SPP模型检测出一幅图像中的多个人物，然后将这些人物图像送入到FastPose模型中进行姿态估计。
我们对YOLOv3_SPP模型以及FastPose模型都进行了加速， 并记录了加速前后的mAP值 (COCO val 2017， Tesla T4)。 其中ground truth box表示FastPose模型
的检测精度， detection boxes表示YOLOv_SPP + FastPose模型的检测结果。
<center>

| Method | ground truth box mAP@0.6 | detection boxes mAP@0.6 | 
|:-------|:-----:|:-------:|
| AlphaPose | 0.743 |0.718 | 
| **AlphaPose_trt** | **0.743** | **0.718** |

</center>


### 2.1 YOLOv3-SPP speed up
下表记录了YOLOv3_SPP模型在不同batch size下的推理时间以及吞吐量，并计算了加速比(第三列以及第四列)。
<center>

| model | Batchsize | Latency (ms) | Throughput  | Latency Speedup |Throughput speedup|
|:-------|:-----:|:-------:|:-----:|:-------:|:-------:|
| YOLOv3-SPP | 1 | 0.0541 | 18484.29 |  |  |
|  | 2 | 0.0939 | 21299.25 |  |  |
|  | 4 | 0.1726 | 23174.97 |  |  |
|  | 8 | 0.3228 | 24783.15 |  |  |
| **YOLOv3-SPP_trt** | 1*| 0.0201 | 49751.24 | **2.7x** | **2.7x** |
|  | 2 | 0.0337 | 59347.18 | **2.8x** | **2.8x** |
|  | 4 | 0.0605 | 66115.70 | **2.9x** | **2.9x** |
|  | 8 | 0.1155 | 69264.07 | **2.8x** | **2.8x** |


</center>

### 2.2 Fast Pose speed up
下面的表格列举了YOLOv3-SPP模型的加速比信息：
<center>

| model | Batchsize | Latency (ms) | Throughput  | Latency Speedup |Throughput speedup|
|:-------|:-----:|:-------:|:-----:|:-------:|:-------:|
| AlphaPose | 1 | 0.0239 | 41841.00 |  |  |
|  | 2 | 0.0246 | 81300.81 |  |  |
|  | 4 | 0.0279 | 143369.17 |  |  |
|  | 8 | 0.0332 | 240963.85 |  |  |
|  | 16 | 0.0566 | 282685.51 |  |  |
|  | 32 | 0.1058 | 302457.46 |  |  |
|  | 64 | 0.2062 | 310378.27 |  |  |
| **AlphaPose_trt** | 1 | 0.00149 | 671140.94 | **16.0x** | **16.0x** |
|  | 2 | 0.00232 | 862068.96 | **10.6x** | **10.6x** |
|  | 4 | 0.00406 | 985221.67 | **6.9x** | **6.9x** |
|  | 8 | 0.00769 | 1040312.09 | **4.3x** | **4.3x** |
|  | 16 | 0.01516 | 1055408.97 | **3.7x** | **3.7x** |
|  | 32 | 0.02998 | 1067378.25 | **3.5x** | **3.5x** |
|  | 64 | 0.05967 | 1072565.78 | **3.5x** | **3.5x** |

</center>

## 3. Code installation
   AlphaPose的安装参考自[这](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md)
#### 3.1 使用conda进行安装

Install conda from [here](https://repo.anaconda.com/miniconda/)
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

#### 3.2 使用pip进行安装
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

## 4. YOLOv3-SPP(PyTorch) to engine
### 4.1 转成static shape的engine模型
(1) YOLOv3_SPP转成onnx模型

下载YOLOv3_SPP[cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg) 以及[weights](https://pjreddie.com/media/files/yolov3-spp.weights) ，并分别放在
./detector/yolo/cfg/以及./detector/yolo/data/文件夹下。
YOLOv3_SPP输入数据的尺寸默认为: 1x3x608x608


``` shell
python ./darknet2onnx.py 
--cfg ./detector/yolo/cfg/yolov3-spp.cfg 
--weight ./detector/yolo/data/yolov3-spp.weights
```
执行该命令之后，会在当前目录下产生一个yolov3_spp_static.onnx模型

(2) 由于YOLOv3-SPP模型中存在Padding操作

trt不能直接识别，因此需要onnx进行修改 [this issue](https://github.com/onnx/onnx-tensorrt/blob/master/docs/faq.md#inputsat0-must-be-an-initializer-or-inputsat0is_weights
)。需要额外下载tensorflow-gpu == 2.4.1以及polygraphy == 0.22.0模块。
``` shell
polygraphy surgeon sanitize yolov3_spp_static.onnx 
--fold-constants 
--output yolov3_spp_static_folded.onnx
```
执行该命令之后，会在当前目录下产生一个yolov3_spp_static_folded.onnx模型

(3) 由onnx模型生成engine

需要注册ScatterND plugin，[this repository](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/plugins)
将plugins以及Makifile放到当前目录下放到当前目录下，然后对make MakeFile文件，进行编译，编译之后会在build文件加下产生
一个ScatterND.so动态库。
``` shell 
trtexec --onnx=yolov3_spp_static_folded.onnx 
--explicitBatch 
--saveEngine=yolov3_spp_static_folded.engine 
--workspace=10240 --fp16 --verbose 
--plugins=build/ScatterND.so
```
执行该命令之后，会在当前目录下产生一个yolov3_spp_static_folded.engine模型

### 4.2 生成dynamic shape的engine模型
(1)  YOLOv3_SPP模型转成onnx模型

输入数据的默认尺寸为: -1x3x608x608 (-1表示batch size可变)
``` shell
python darknet2onnx_dynamic.py 
--cfg ./detector/yolo/cfg/yolov3-spp.cfg 
--weight ./detector/yolo/data/yolov3-spp.weights
```
执行该命令之后，会在当前目录下产生一个yolov3_spp_-1_608_608_dynamic.onnx模型

(2) 对onnx模型就行修改
``` shell
polygraphy surgeon sanitize yolov3_spp_-1_608_608_dynamic.onnx 
--fold-constants 
--output yolov3_spp_-1_608_608_dynamic_folded.onnx
```

(3) 由onnx模型转成engine

minShapes设置能够输入数据的最小尺寸，optShapes可以与minShapes保持一致，maxShapes设置
输入数据的最大尺寸，这三个是必须要设置的，可通过trtexec -h查看具体用法。
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
执行该命令之后，会在当前目录下产生一个yolov3_spp_-1_608_608_dynamic_folded.engine 模型

## 5. FastPose to engine
### 5.1 生成static shape的engine模型
(1) FastPose转成onnx模型

模型输入数据的默认尺寸为: 1x3x256x192
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

### 5.2 生成dynamic shape的engine模型
(1) 生成onnx模型

模型输入数据的默认尺寸为：-1x3x256x192 (-1表示batch size可变)
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


## 6. Inference
这一部分主要使用两个加速模型对图像以及视频进行检测
### 6.1 对图像进行检测
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

### 6.2 对视频进行检测
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


## 7. Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe，
      title={{RMPE}: Regional Multi-person Pose Estimation}，
      author={Fang， Hao-Shu and Xie， Shuqin and Tai， Yu-Wing and Lu， Cewu}，
      booktitle={ICCV}，
      year={2017}
    }

    @article{li2018crowdpose，
      title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark}，
      author={Li， Jiefeng and Wang， Can and Zhu， Hao and Mao， Yihuan and Fang， Hao-Shu and Lu， Cewu}，
      journal={arXiv preprint arXiv:1812.00324}，
      year={2018}
    }

    @inproceedings{xiu2018poseflow，
      author = {Xiu， Yuliang and Li， Jiefeng and Wang， Haoyu and Fang， Yinghong and Lu， Cewu}，
      title = {{Pose Flow}: Efficient Online Pose Tracking}，
      booktitle={BMVC}，
      year = {2018}
    }


## 8. Reference

(1) [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)

(2) [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)

(3) [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)



