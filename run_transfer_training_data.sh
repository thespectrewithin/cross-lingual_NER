#!/bin/bash

# Replace the paths below with your own
WORD_TRANSLATION_FILE=csls_eng_esp_fasttext_idc
SOURCE_TRAINING_DATA=conll2003/eng.train_iob2
OUTPUT_FILE=conll2002/esp.train_transferred_fasttext_idc

python transfer_training_data.py --word_translation $WORD_TRANSLATION_FILE --training_data $SOURCE_TRAINING_DATA --output $OUTPUT_FILE --same_script
