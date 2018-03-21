#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=1         # number of parallel jobs - 1 is perfect for such a small data set
#lm_order=1     # language model order (n-gram quantity) - 1 is enough for digits grammar
#nt=13
# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }

# Removing previously created data (from last run.sh execution)
# rm -rf exp mfcc data/train/spk2utt data/train/utt2dur data/train/cmvn.scp data/train/feats.scp data/train/split1 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/local/lang data/lang data/local/tmp data/local/dict/lexiconp.txt
rm -rf  mfcc/cmvn_test.scp mfcc/cmvn_test.ark mfcc/raw_mfcc_test.1.ark mfcc/raw_mfcc_test.1.scp data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split*


# Making spk2utt files
# utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

# Making feats.scp files
mfccdir=mfcc


utils/validate_data_dir.sh --no-feats --no-text data/test     # script for checking if prepared data is all right
utils/fix_data_dir.sh data/test          # tool for data sorting if something goes wrong above


steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test $mfccdir
#
# # Making cmvn.scp files
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir


echo
echo "===== MONO DECODING ====="
echo

steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode

echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo

steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode

echo
echo "===== TRI2A DECODING ====="
echo

steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri2a/graph data/test exp/tri2a/decode || exit 1

echo
echo "===== TRI2A MMI DECODING ====="
echo

steps/decode.sh --config conf/decode.config --iter 4 --nj 1 --cmd "$decode_cmd" exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4

echo
echo "===== TRI2A MPE DECODING ====="
echo

steps/decode.sh --config conf/decode.config --iter 4 --nj 1 --cmd "$decode_cmd" exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4

echo
echo "===== TRI2B DECODING ====="
echo

steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri2b/graph data/test exp/tri2b/decode

echo
echo "===== Tri3b [LDA+MLLT+SAT] DECODING ====="
echo

steps/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode


echo
echo "===== Raw_fMLLR  DECODING ====="
echo

#steps/decode_raw_fmllr.sh  --config conf/decode.config --nj 1 --cmd "$decode_cmd" \
#       exp/tri3c/graph data/test exp/tri3c/decode


echo
echo "===== Normal_fMLLR  DECODING ====="
echo

#steps/decode_raw_fmllr.sh --use-normal-fmllr true --config conf/decode.config --nj 1 --cmd "$decode_cmd" \
#      exp/tri3c/graph data/test exp/tri3c/decode_2fmllr

echo
echo "===== SGMM  DECODING ====="
echo

steps/decode_sgmm2.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/sgmm2/graph data/test exp/sgmm2/decode || exit 1


steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/sgmm2/graph data/test exp/sgmm2/decode_fmllr || exit 1

# echo
# echo "===== DNN  DECODING ====="
# echo

# utils/mkgraph.sh data/lang exp/tri4_DNN exp/tri4_DNN/graph || exit
# steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 1 --feat-type raw --transform-dir exp/tri4_DNN/decode exp/tri4_DNN/graph data/test exp/tri4_DNN/decode_it4 | tee exp/tri4_DNN/decode_it4/decode.log || exit 



# local/run_raw_fmllr.sh
# local/nnet2/run_4a.sh
#local/run_sgmm2.sh

echo
echo "===== run.sh script is finished ====="
echo
