#!/bin/bash

# Choose your device number
DEVICE=1

# Replace the paths below with your own
ESP_BASE_PATH=./conll2002/esp
ESP_TRAIN_PATH=$ESP_BASE_PATH.train_transferred_fasttext_idc
ESP_DEV_PATH=$ESP_BASE_PATH.testa
ESP_TEST_PATH=$ESP_BASE_PATH.testb

ESP_WORD_VEC_PATH=./spanish.glove.gigaword_wiki.100d.txt

LOG_PATH=./esp_fasttext_idc_sa_result

for SEED in 10 21 1001 2112 3223; do
    echo "$ THEANO_FLAGS=device=cuda$DEVICE python lasagne_ncrf.py --train_data $ESP_TRAIN_PATH --dev_data $ESP_DEV_PATH --test_data $ESP_TEST_PATH --word_vec_path $ESP_WORD_VEC_PATH --seed $SEED --word_dropout 0.5 --use_crf --self_att --att_dropout 0.5 --log_interval 150 --log_path $LOG_PATH"
    THEANO_FLAGS=device=cuda$DEVICE python lasagne_ncrf.py --train_data $ESP_TRAIN_PATH --dev_data $ESP_DEV_PATH --test_data $ESP_TEST_PATH --word_vec_path $ESP_WORD_VEC_PATH --seed $SEED --word_dropout 0.5 --use_crf --self_att --att_dropout 0.5 --log_interval 150 --log_path $LOG_PATH
done
