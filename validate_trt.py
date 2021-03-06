"""Validation script."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
from alphapose.utils.pPose_nms import oks_pose_nms
from tools.trt_lite import TrtTiny
import pycuda

parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    default='./pretrained_models/fast_res50_256x192.pth',
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default=1,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo_trt")
parser.add_argument('--pose_engine', default='./fastPose.engine', help='the path of pose engine')
parser.add_argument('--yolo_engine', default='./yolov3_spp_static_folded.engine', help='the path of yolo engine')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size PER GPU')
parser.add_argument('--dll_file', default='./build/ScatterND.so', help='the dll file path')
parser.add_argument('--out_height', default=22743, type=int, help='the yolo output height')
parser.add_argument('--out_width', default=85, type=int, help='the yolo output width')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def validate(pose_model, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    # m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output = pose_model.detect_context(inps.cpu().numpy())
        output = torch.from_numpy(output).to(opt.device)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp).cuda() for inp in inps]
            else:
                inps_flip = flip(inps).cuda()

            # ?????????????????????
            inps_flip = pose_model.detect_context(inps_flip.cpu().numpy())
            inps_flip = torch.from_numpy(inps_flip).to(opt.device)
            output_flip = flip_heatmap(inps_flip, det_dataset.joint_pairs, shift=True)
            # output_flip = flip_heatmap(m(inps_flip), det_dataset.joint_pairs, shift=True)
            pred_flip = output_flip[:, eval_joints, :, :]
        else:
            output_flip = None

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox, hms_flip=pred_flip[i], hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['score'] = float(scores[i])
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    kpt_json = oks_pose_nms(kpt_json)

    with open('./exp/json/trt/validate_rcnn_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/trt/validate_rcnn_kpt.json', ann_type='keypoints',
                       ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


def validate_gt(pose_model, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    # m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        # ??????static shape???fast pose???engine????????????
        output = pose_model.detect_context(inps.cpu().numpy())
        # ???????????????cuda???????????????????????????
        output = torch.from_numpy(output).to(opt.device)
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp).cuda() for inp in inps]
            else:
                inps_flip = flip(inps).cuda()
            inps_flip = pose_model.detect_context(inps_flip.cpu().numpy())
            inps_flip = torch.from_numpy(inps_flip).to(opt.device)
            # output_flip = flip_heatmap(m(inps_flip), gt_val_dataset.joint_pairs, shift=True)
            output_flip = flip_heatmap(inps_flip, gt_val_dataset.joint_pairs, shift=True)
            pred_flip = output_flip[:, eval_joints, :, :]
        else:
            output_flip = None

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i], bbox, hms_flip=pred_flip[i], hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open('./exp/json/trt/validate_gt_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP('./exp/json/trt/validate_gt_kpt.json', ann_type='keypoints',
                       ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


if __name__ == "__main__":
    # # ????????????
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    # ??????trt?????????
    pose_model = TrtTiny(batch_size=opt.batch, out_height=opt.out_height, out_width=opt.out_width,
                         engine_path=opt.pose_engine,
                         cuda_ctx=pycuda.autoinit.context)

    with torch.no_grad():
        # ??????alpha pose???AP???
        gt_AP = validate_gt(pose_model, cfg, heatmap_to_coord, opt.batch)
        print("gt ap = {}".format(gt_AP))
        # ??????yolo?????????
        detbox_AP = validate(pose_model, heatmap_to_coord, opt.batch)
        print("detbox ap = {}".format(detbox_AP))
        print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))

# python validate_trt.py --flip-test
