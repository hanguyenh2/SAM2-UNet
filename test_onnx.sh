#!/bin/bash

CUDA_VISIBLE_DEVICES="" \
python test_onnx.py \
--size 960 \
--checkpoint "/Users/hhn21/Documents/h2/interior/wall_seg_sam2unet_960_20260325.onnx" \
--save_path "/Users/hhn21/Documents/h2/ANDERSEN/data/andersen_crop/wall_masks/" \
--test_image_path "/Users/hhn21/Documents/h2/ANDERSEN/data/andersen_crop/images/" \
--test_gt_path ""
