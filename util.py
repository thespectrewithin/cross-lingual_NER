from random import randint, shuffle
import numpy as np
import pickle as cPickle
from collections import defaultdict
import constants as Constants

def create_char_index(paths, pad=False):

    char_dict = {}
    if pad:
        char_dict[Constants.PAD_WORD] = Constants.PAD
        char_dict[Constants.UNK_WORD] = Constants.UNK
    else:
        char_dict[Constants.UNK_WORD] = 0

    for path in paths:
        for line in open(path):
            l = line.strip().split()
            if len(l) > 1:
                word = l[0]
                for i in range(len(word)):
                    if word[i] not in char_dict:
                        char_dict[word[i]] = len(char_dict)

    return char_dict

def create_word_index(path, vocab_size, wembed_size, pad=False):

    word_dict = {}
    if pad:
        word_dict[Constants.PAD_WORD] = Constants.PAD
        word_dict[Constants.UNK_WORD] = Constants.UNK
    else:
        word_dict[Constants.UNK_WORD] = 0

    word_vector = []

    num = 0

    lower = -np.sqrt(3.0 / wembed_size)
    upper = np.sqrt(3.0 / wembed_size)
    unk_vec = np.asarray(np.random.uniform(lower, upper, size=(wembed_size, )), dtype='float32')
    if pad:
        word_vector.append(np.zeros((wembed_size, )))
    word_vector.append(unk_vec)
    
    for line in open(path):
        if num < vocab_size:
            l = line.strip().split()
            word_dict[l[0]] = len(word_dict)
            vec = np.array([float(i) for i in l[1:]], dtype = 'float32')
            word_vector.append(vec)
        num += 1

    return word_dict, np.vstack(word_vector)

def iob2(tags):

    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def load_data(path, label_index, isIOB2):
    
    sentences = []
    labels = []

    sentence = []
    label = []
    for line in open(path, 'r'):
        if len(line.strip()) != 0:
            sentence.append(line.strip().split()[0])
            label.append(line.strip().split()[label_index])
        else:
            if len(sentence) > 1:
                sentences.append(sentence)
                if not isIOB2:
                    iob2(label)
                labels.append(label)
                sentence = []
                label = []
    if len(sentence) > 1:
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def iterate_minibatches(inputs1, inputs2, batch_size, shuffle=False):
    
    batch_num = int(np.ceil(len(inputs1) / float(batch_size)))
    if shuffle:
        rand_indices = np.random.permutation(len(inputs1))

    for cur_batch in range(0, batch_num):
        cur_batch_size = batch_size if cur_batch < batch_num - 1 else len(inputs1) - batch_size * cur_batch
        if shuffle:
            yield [inputs1[i] for i in rand_indices[cur_batch * batch_size : cur_batch * batch_size + cur_batch_size]], [inputs2[i] for i in rand_indices[cur_batch * batch_size : cur_batch * batch_size + cur_batch_size]]
        else:
            yield inputs1[cur_batch * batch_size : cur_batch * batch_size + cur_batch_size], inputs2[cur_batch * batch_size : cur_batch * batch_size + cur_batch_size]

def get_entity(label):
    entities = []
    i = 0
    while i < len(label):
        if label[i] != 'O':
            e_type = label[i][2:]
            j = i + 1
            while j < len(label) and label[j] == 'I-' + e_type:
                j += 1
            entities.append((i, j, e_type))
            i = j
        else:
            i += 1
    return entities

def evaluate_ner(pred, gold):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        pred_entities = get_entity(pred[i])
        gold_entities = get_entity(gold[i])
        temp = 0
        for entity in pred_entities:
            if entity in gold_entities:
                tp += 1
                temp += 1
            else:
                fp += 1
        fn += len(gold_entities) - temp
    precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0 
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# def save_model(params, epoch):
#     print('saving model....')
#     eout = open('./eng_model/en_' +  `epoch`, 'w')
#     cPickle.dump(params, eout, cPickle.HIGHEST_PROTOCOL)
#     print('done')
#     eout.close()

# def load_model(epoch):
#     print('loading model...')
#     ein = open('./eng_model/best_models/en_' + `epoch`)
#     params = cPickle.load(ein)
#     ein.close()
#     print('done')
#     return params
