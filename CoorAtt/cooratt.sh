#!/usr/bin/env bash
GPU_ID=0
#data_dir=/data/jindwang/office31
## Office31
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee BNM_D2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee BNM_D2W.log
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee BNM_A2D.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee BNM_A2W.log
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee BNM_W2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee BNM_W2D.log


data_dir=/DB/rhome/yuhaowang/CV_DA/OfficeHomeDataset_10072016

#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart
# Office-Home
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee ./CoorAtt/Coor_1ATT_A2C.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee ./CoorAtt/Coor_1ATT_A2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee ./CoorAtt/Coor_1ATT_A2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee ./CoorAtt/Coor_1ATT_C2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee ./CoorAtt/Coor_1ATT_C2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee ./CoorAtt/Coor_1ATT.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee ./CoorAtt/Coor_1ATT_P2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee ./CoorAtt/Coor_1ATT_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee ./CoorAtt/Coor_1ATT_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee ./CoorAtt/Coor_1ATT_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee ./CoorAtt/Coor_1ATT_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config CoorAtt/cooratt.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee ./CoorAtt/Coor_1ATT_R2C.log
