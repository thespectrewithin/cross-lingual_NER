# Neural Cross-Lingual Named Entity Recognition with Minimal Resources

This is the code we used in our paper
>[Neural Cross-Lingual Named Entity Recognition with Minimal Resources](https://arxiv.org/abs/1808.09861)

>Jiateng Xie, Zhilin Yang, Graham Neubig, Noah A. Smith, Jaime Carbonell

>EMNLP 2018

## Requirements

Python 2.7 or 3.6
PyTorch 0.3.0 or 0.4.0
Theano 1.0
Lasagne 0.2

The original results on the paper are tuned and obtained using the NER model written in Theano/Lasagne. Everything else is in PyTorch. We also provide a PyTorch implementation of the NER model, which might produce slightly worse results, due to implementation differences between the library such as different weight initialization.

## Train Bilingual Word Embeddings

To train bilingual word embeddings, we use [MUSE](https://github.com/facebookresearch/MUSE).

After installing MUSE, to get a mapping (e.g., en-es, identical character strings), first set ``VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'`` in ``supervised.py``, and then run:
``python supervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec  --n_refinement 3 --dico_train identical_char --max_vocab 100000``
which will produce a mapping at a location such as ``/your_path/MUSE/dumped/debug/qbun3algl8/best_mapping.pth``

To create a word-to-word translation file, run:
``./run_load_muse.sh``
Note, if your embedding file contains a 1st line that specifies the size and the dimension of the embedding, such as ``2519370 300``, remove it before you run this script (include it though when running MUSE).

# Data Format

We use IOB2 tagging scheme, and NER data in the following format:
``Peter B-PER``
``Blackburn I-PER``

## Transfer Training Data

Simply run:
``./run_transfer_training_data.sh``

## Train Cross-Lingual NER Model

For the Lasagne/Theano implementation, run:
``run_lasagne_ncrf.sh``

For the PyTorch implementation, run:
``run_pytorch_ncrf.sh``
