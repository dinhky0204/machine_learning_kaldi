#!/bin/bash

. ./cmd.sh
. ./path.sh || exit 1

dnn_mem_reqs="mem_free=4.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
dnn_mem_reqs="mem_free=4.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.02 --final-learning-rate 0.004 --num-hidden-layers 2 --num-jobs-nnet 2 --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" --num-threads 2 --samples-per-iter 200000 data/train data/lang exp/tri3b_ali exp/tri4_DNN
