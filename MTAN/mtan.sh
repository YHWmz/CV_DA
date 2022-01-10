#!/usr/bin/env bash
GPU_ID=3
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

CUDA_VISIBLE_DEVICES=$GPU_ID python mtan.py --config MTAN/mtan.yaml --data_dir $data_dir --src_domain Art --tgta_domain Clipart --tgtb_domain Product --tgtc_domain Real_World
# # Office-Home
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee BNM_A2C.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee Transfer_ATT_A2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee BNM_A2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee BNM_C2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee Transfer_ATT_C2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee BNM_C2P.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee BNM_P2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python cooratt.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee Transfer_ATT_P2R.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee BNM_P2C.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Art | tee BNM_R2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Product | tee BNM_R2P.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart | tee BNM_R2C.log