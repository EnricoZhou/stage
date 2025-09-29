#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

# source activate ct-2

# AGGIUNTI PER TESTARE
export NUM_SPLITS=${NUM_SPLITS:-10}   # 👈 ridotto per test locale, puoi rimettere 10 se vuoi

# AGGIUNTO
# export NUM_SPLITS=2
# export TREATMENT=buzzy_title


export INIT_DIR=../pre-trained/uncased_L-12_H-768_A-12
export INIT_FILE=$INIT_DIR/bert_model.ckpt
export BERT_BASE_DIR=../pre-trained/uncased_L-12_H-768_A-12
#export INIT_FILE=$BERT_BASE_DIR/bert_model.ckpt
export DATA_FILE=../dat/PeerRead/proc/arxiv-all.tf_record
# export OUTPUT_DIR=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split${SPLIT}
# export PREDICTION_FILE=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split${SPLIT}/predictions.tsv

# export OUTPUT_DIR=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split0
# export PREDICTION_FILE=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split0/predictions.tsv


# export INIT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/pre-training/PeerRead/pretrained
# export INIT_FILE=$INIT_DIR/bert_model.ckpt-102
# export BERT_BASE_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12
# #export INIT_FILE=$BERT_BASE_DIR/bert_model.ckpt
# export DATA_FILE=/proj/sml_netapp/dat/undocumented/PeerRead/proc/arxiv-all.tf_record
# #export OUTPUT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/cb_test
# export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

OUTPUT_DIR_BASE=../out/PeerRead/real/
export TREATMENT=${TREATMENT:-buzzy_title}
export SPLIT=${SPLIT:-0}
export OUTPUT_DIR=${OUTPUT_DIR_BASE}o_accepted_t_${TREATMENT}/split${SPLIT}
export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

echo "python -m PeerRead.model.run_causal_bert \
  --seed=${SPLIT} \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=32 \
  --learning_rate=3e-4 \
  --num_train_epochs=20 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
  --do_masking=True \
  --simulated=real \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --treatment=${TREATMENT}
"

python -m PeerRead.model.run_causal_bert \
  --seed=${SPLIT} \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=32 \
  --num_train_epochs=20 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
  --do_masking=True \
  --simulated=real \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --treatment=${TREATMENT}

# --strategy_type=mirror \


