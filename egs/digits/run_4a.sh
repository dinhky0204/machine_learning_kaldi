#!/bin/bash

. ./cmd.sh
. ./path.sh || exit 1
export LC_ALL=C

steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--allow-large-dim=true" 1800 9000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1
