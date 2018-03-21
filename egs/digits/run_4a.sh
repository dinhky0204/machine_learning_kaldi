#!/bin/bash

. ./cmd.sh
. ./path.sh || exit 1
export LC_ALL=C

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
# steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 --final-learning-rate 0.002 --num-hidden-layers 2 --num-jobs-nnet $njobs --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" data/train data/lang exp/tri3_ali exp/tri4_DNN
steps/nnet2/train_tanh.sh data/train data/lang exp/tri3b_ali exp/tri4_nnet
