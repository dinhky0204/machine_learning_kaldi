#!/bin/bash

. ./cmd.sh
. ./path.sh || exit 1
export LC_ALL=C

steps/decode.sh --config conf/decode.config --nj 1 --cmd "$decode_cmd" exp/tri2a/graph data/test exp/tri2a/decode || exit 1
