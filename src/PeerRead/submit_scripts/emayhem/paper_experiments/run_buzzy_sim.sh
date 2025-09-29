#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

# COMMENTATO
# source activate ct-2

# # AGGIUNTI PER TESTARE
# export NUM_SPLITS=2   # 👈 ridotto per test locale, puoi rimettere 10 se vuoi
# export BETA0=0.25
# export BETA1=1.0
# export GAMMA=0.0
# export SIMMODE=simple


# export INIT_DIR=../pre-trained/uncased_L-12_H-768_A-12
# export INIT_FILE=$INIT_DIR/bert_model.ckpt
# export BERT_BASE_DIR=../pre-trained/uncased_L-12_H-768_A-12
# #export INIT_FILE=$BERT_BASE_DIR/bert_model.ckpt
# export DATA_FILE=../dat/PeerRead/proc/arxiv-all.tf_record
# # export OUTPUT_DIR=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split${SPLIT}
# # export PREDICTION_FILE=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split${SPLIT}/predictions.tsv

# export OUTPUT_DIR=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split0
# export PREDICTION_FILE=../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0/split0/predictions.tsv

# Variabili: se non sono già impostate da prova.sh, diamo default
export NUM_SPLITS=${NUM_SPLITS:-2}
export BETA0=${BETA0:-0.25}
export BETA1=${BETA1:-1.0}
export GAMMA=${GAMMA:-0.0}
export SIMMODE=${SIMMODE:-simple}
export SPLIT=${SPLIT:-0}

export INIT_DIR=../pre-trained/uncased_L-12_H-768_A-12
export INIT_FILE=$INIT_DIR/bert_model.ckpt
export BERT_BASE_DIR=../pre-trained/uncased_L-12_H-768_A-12
export DATA_FILE=../dat/PeerRead/proc/arxiv-all.tf_record

# Usa OUTPUT_DIR e PREDICTION_FILE in base a SPLIT
export OUTPUT_DIR=../out/PeerRead/buzzy-based-sim/mode${SIMMODE}/beta0${BETA0}.beta1${BETA1}.gamma${GAMMA}/split${SPLIT}
export PREDICTION_FILE=${OUTPUT_DIR}/predictions.tsv
mkdir -p "${OUTPUT_DIR}"

echo ">>> Eseguo split ${SPLIT}, output in ${OUTPUT_DIR}"

echo "python -m PeerRead.model.run_causal_bert \
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=4 \
  --learning_rate=5e-4 \
  --num_train_epochs=1 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
  --do_masking=True \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --simulated=attribute \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --simulation_mode=${SIMMODE}"

python -m PeerRead.model.run_causal_bert \
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=4 \
  --num_train_epochs=1 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
  --do_masking=True \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --simulated=attribute \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --simulation_mode=${SIMMODE}