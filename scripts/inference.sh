set -x

CONFIG=$1
CKPT=$2
VIDEO=$3
OUTDIR=${4:-"./examples/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video


# ./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
# pretrained_models/fast_res50_256x192.pth ./videos/blCode_action1_scene1.avi