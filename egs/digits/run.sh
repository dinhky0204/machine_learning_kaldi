#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1
export LC_ALL=C

nj=1       # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar
nt=1

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; } 

# Removing previously created data (from last run.sh execution)
rm -rf exp mfcc data/train/spk2utt data/train/cmvn.scp data/train/feats.scp data/train/split1 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/local/lang data/lang data/local/tmp data/local/dict/lexiconp.txt

echo
echo "===== PREPARING ACOUSTIC DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts): 
#
# spk2gender  [<speaker-id> <gender>]
# wav.scp     [<uterranceID> <full_path_to_audio_file>]
# text	      [<uterranceID> <text_transcription>]
# utt2spk     [<uterranceID> <speakerID>]
# corpus.txt  [<text_transcription>]

# Making spk2utt files
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

echo
echo "===== FEATURES EXTRACTION ====="
echo

# Making feats.scp files
mfccdir=mfcc
steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 $x exp/make_mfcc/$x $mfccdir --mfcc-config conf/mfcc.conf
steps/compute_cmvn_stats.sh $x exp/make_mfcc/$x $mfccdir
# Uncomment and modify arguments in scripts below if you have any problems with data sorting
utils/validate_data_dir.sh data/train     # script for checking prepared data - here: for data/train directory
utils/fix_data_dir.sh data/train          # tool for data proper sorting if needed - here: for data/train directory
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test $mfccdir

# Making cmvn.scp files
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir

echo
echo "===== PREPARING LANGUAGE DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts): 
#
# lexicon.txt           [<word> <phone 1> <phone 2> ...]		
# nonsilence_phones.txt	[<phone>]
# silence_phones.txt    [<phone>]
# optional_silence.txt  [<phone>]

# Preparing language data
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo

loc=`which ngram-count`;
if [ -z $loc ]; then
 	if uname -a | grep 64 >/dev/null; then
		sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
	else
    		sdir=$KALDI_ROOT/tools/srilm/bin/i686
  	fi
  	if [ -f $sdir/ngram-count ]; then
    		echo "Using SRILM language modelling tool from $sdir"
    		export PATH=$PATH:$sdir
  	else
    		echo "SRILM toolkit is probably not installed.
		      Instructions: tools/install_srilm.sh"
    		exit 1
  	fi
fi

local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa

echo
echo "===== MAKING G.fst ====="
echo

lang=data/lang
cat $local/tmp/lm.arpa | arpa2fst - | fstprint | utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon | fstarcsort --sort_type=ilabel > $lang/G.fst

echo
echo "===== MONO TRAINING ====="
echo

steps/train_mono.sh --boost-silence 1.5 --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1

echo
echo "===== MONO ALIGNMENT =====" 
echo

steps/align_si.sh --boost-silence 1.5 --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1


utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode



# for tri1 training
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo

steps/train_deltas.sh --boost-silence 1.5 --cmd "$train_cmd" 2500 15000 data/train data/lang exp/mono_ali exp/tri1 || exit 1

# for tri1 decoding
echo
echo "===== TRI1 ALIGNMENT ====="
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali || exit 1

utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1
steps/decode.sh --config conf/decode.config --nj 1 --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode

# for tri2a training
echo 
echo "===== TRI2A TRAINING ====="
echo 

steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1
utils/mkgraph.sh data/lang exp/tri2a exp/tri2a/graph
steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri2a exp/tri2a_ali || exit 1

# echo
# echo "===== TRI2A DECODE ====="
# echo 

# steps/decode.sh --scoring-opts "--min-lmw $min_lmw --max-lmw $max_lmw" --config common/decode.conf --nj $njobs --cmd "$decode_cmd" $EXP/tri2a/graph_
# ${lm} $WORK/$tgt_dir $EXP/tri2a/decode_${tgt_dir}
# steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri2a/graph data/test exp/tri2a/decode || exit 1

# for tri2b training
echo
echo "===== TRANING TRI2B ====="
echo

steps/train_lda_mllt.sh --cmd "$train_cmd" 1800 9000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1
steps/train_lda_mllt.sh --cmd "$train_cmd" --allow-large-dim true 1800 9000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1

echo
echo "===== TRI2B ALIGNMENT ====="
echo

steps/align_si.sh --nj $nt --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_ali || exit 1

# for train tri2b_mmi
# echo
# echo "===== TRAINING TRI2B_MMI ====="
# echo
# steps/make_denlats.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1
# steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi || exit 1

# for training tri2b_bmmi
# steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b || exit 1
# steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1

# for tri3b training
# echo
# echo "===== TRANING TRI3B ====="
# echo

# steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1

# echo
# echo "===== ALIGMENT TRI3B ====="
# echo
# steps/align_fmllr.sh --nj $nt --cmd "$train_cmd" data/train data/lang exp/tri3b exp/tri3b_ali || exit 1

#for training triphone SGMM
# echo
# echo "===== TRAINING TRI SGMM ====="
# echo
# steps/train_ubm.sh --cmd "$train_cmd" 200 data/train data/lang exp/tri3b_ali exp/ubm4
# steps/train_sgmm2.sh --cmd "$train_cmd" 7000 9000 data/train data/lang exp/tri3b_ali exp/ubm4/final.ubm exp/sgmm2

# for dnn training, run the follow scripts
# echo
# echo "===== DNN TRAINING ====="
# echo

# dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
# dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
# # steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 --final-learning-rate 0.002 --num-hidden-layers 2 --num-jobs-nnet $njobs --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" data/train data/lang exp/tri3_ali exp/tri4_DNN
# steps/nnet2/train_tanh.sh data/train data/lang exp/tri3b_ali exp/tri4_nnet

# echo
# echo "TRAINING DNN"
# echo

#dnn_mem_reqs="mem_free=4.0G,ram_free=0.2G"
#dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
# steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.02 --final-learning-rate 0.004 --num-hidden-layers 3 --num-jobs-nnet 1 --cmd #"$train_cmd" "${dnn_train_extra_opts[@]}" --num-threads 1 data/train data/lang exp/tri3b_ali exp/tri4_DNN

# local/run_raw_fmllr.sh
# local/nnet2/run_4a.sh

# echo
# echo "===== run.sh script is finished ====="
# echo
