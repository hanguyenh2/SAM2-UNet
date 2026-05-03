#!/bin/bash

python eval.py \
--pred_path "../results_SAM2-UNet_epoch-114_loss-0.483_iou-0.935/" \
--gt_path "/Users/hhn21/Documents/h2/ANDERSEN/andersen_boundary_10.3/data_test/masks/"
