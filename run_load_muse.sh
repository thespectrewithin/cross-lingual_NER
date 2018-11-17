#!/bin/bash

# the embedding file it expects doesn't contain the 1st line that specifies the size and the dimension of the embedding file usually in fasttext embeddings

#Choose your device number
DEVICE=2

# Replace the paths below with your own
ENG_PATH=wiki.en.vec
ESP_PATH=wiki.es.vec
O_PATH=/your_path/MUSE/dumped/debug/ce80cim2hl(replace this)/best_mapping.pth
OUTPUT_PATH=csls_eng_esp_fasttext_idc

echo "$ CUDA_VISIBLE_DEVICES=$DEVICE python load_muse.py --eng_path $ENG_PATH --esp_path $ESP_PATH --o_path $O_PATH --output_path $OUTPUT_PATH"
CUDA_VISIBLE_DEVICES=$DEVICE python load_muse.py --eng_path $ENG_PATH --esp_path $ESP_PATH --o_path $O_PATH --output_path $OUTPUT_PATH
