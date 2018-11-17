from random import randint
import argparse
import sys
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time
from lasagne_crf import CRFLayer
from lasagne_self_att_layer import self_att_layer, dotlayer
from lasagne_objectives import crf_loss, crf_accuracy
from collections import defaultdict
from util import create_char_index, create_word_index, load_data, evaluate_ner, iterate_minibatches

CAT = ['PER', 'ORG', 'LOC', 'MISC']
POSITION = ['I', 'B']

LABEL_INDEX = ['O'] + ["{}-{}".format(position, cat) for cat in CAT for position in POSITION]

def to_input_variable(data, label, word_vocab, char_vocab):

    word_max_seq_len = max(len(s) for s in data)
    char_max_seq_len = max(len(s) for w in data for s in w)

    word_input = np.zeros((len(data), word_max_seq_len), dtype='int32')
    word_input_mask = np.zeros((len(data), word_max_seq_len), dtype='float32')

    char_input = np.zeros((len(data), word_max_seq_len, char_max_seq_len), dtype='int32')
    char_input_mask = np.zeros((len(data), word_max_seq_len, char_max_seq_len), dtype='float32')

    label_input = np.zeros((len(data), word_max_seq_len), dtype='int32')

    for i in range(len(data)):
        word_input_mask[i][:len(data[i])] = 1
        for j in range(len(data[i])):
            if data[i][j].lower() in word_vocab:
                word_input[i][j] = word_vocab[data[i][j].lower()]
            label_input[i][j] = LABEL_INDEX.index(label[i][j])
            char_input_mask[i][j][:len(data[i][j])] = 1
            for k in range(len(data[i][j])):
                c = data[i][j][k]
                if c in char_vocab:
                    char_input[i][j][k] = char_vocab[c if not c.isdigit() else '0']

    return word_input, word_input_mask, char_input, char_input_mask, label_input

def main():

    parser = argparse.ArgumentParser(description='lasagne simple neural crf')

    parser.add_argument('--wembed_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--cembed_size', type=int, default=25, help='char embedding size')
    parser.add_argument('--word_hidden', type=int, default=200, help='number of hidden units word layer')
    parser.add_argument('--char_hidden', type=int, default=50, help='number of hidden units char layer')
    parser.add_argument('--vocab_size', type=int, default=100000, help='vocab size')

    parser.add_argument('--embed_dropout', type=float, default=0.5, help='embed level dropout rate')
    parser.add_argument('--word_dropout', type=float, default=0.5, help='word level dropout rate')
    parser.add_argument('--att_sm_dropout', type=float, default=0.0, help='att softmax level dropout rate')
    parser.add_argument('--att_dropout', type=float, default=0.0, help='att level dropout rate')

    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--nepoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='learning rate decay')
    parser.add_argument('--clip', type=float, default=5.0, help='grad clipping')

    parser.add_argument('--seed', type=int, default=1001, help='random seed')

    parser.add_argument('--finetune', action='store_true', help='fine tune word embedding')
    parser.add_argument('--use_crf', action='store_true',help='use crf')
    parser.add_argument('--self_att', action='store_true',help='use self attention')

    parser.add_argument('--train_data', type=str, help='training data path')
    parser.add_argument('--dev_data', type=str, help='development data path')
    parser.add_argument('--test_data', type=str, help='test data path')
    parser.add_argument('--word_vec_path', type=str, help='word vector path')

    parser.add_argument('--log_interval', type=int, default=0, help='')
    parser.add_argument('--log_path', type=str, default='', help='records best test results for multiple runs')

    #parser.add_argument('--src_vec_path', type=str, help='source word embedding path')
    #parser.add_argument('--tgt_vec_path', type=str, help='target word embedding path')
    
    args = parser.parse_args()

    word_embed_size = args.wembed_size
    char_embed_size = args.cembed_size
    word_hidden_size = args.word_hidden
    char_hidden_size = args.char_hidden
    vocab_size = args.vocab_size

    batch_size = args.batch_size
    num_epoch = args.nepoch
    lr = args.lr
    decay = args.lr_decay
    clip = args.clip

    embed_dropout = args.embed_dropout
    word_dropout = args.word_dropout
    att_sm_dropout = args.att_sm_dropout
    att_dropout = args.att_dropout

    finetune = args.finetune
    use_crf = args.use_crf
    self_att = args.self_att

    log_interval = args.log_interval

    seed = args.seed

    train_data_path = args.train_data
    dev_data_path = args.dev_data
    test_data_path = args.test_data
    word_vec_path = args.word_vec_path
    # src_word_vec_path = args.src_vec_path
    # tgt_word_vec_path = args.tgt_vec_path

    #np.random.seed(rseed)

    lasagne.random.set_rng(np.random)
    np.random.seed(seed)

    print('loading data...')

    char_dict = create_char_index([train_data_path, dev_data_path, test_data_path])
    word_dict, word_vector = create_word_index(word_vec_path, vocab_size, word_embed_size)

    train_X, train_Y = load_data(train_data_path, 1, True)
    dev_X, dev_Y = load_data(dev_data_path, 1, True)
    test_X, test_Y = load_data(test_data_path, 1, True)

    print('building model...')

    word_input_var = T.imatrix()
    char_input_var = T.itensor3()
    char_mask_var = T.tensor3(dtype = 'float32')
    target_var = T.imatrix()
    mask_var = T.matrix(dtype = 'float32')

    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    basize, seq_len = word_input_var.shape

    char_in = lasagne.layers.InputLayer(shape=(None, None, None), input_var = char_input_var)
    char_embed = lasagne.layers.EmbeddingLayer(char_in, len(char_dict), char_embed_size)
    char_embed = lasagne.layers.ReshapeLayer(char_embed, (-1, [2], [3]))

    char_mask = lasagne.layers.InputLayer(shape=(None, None, None), input_var = char_mask_var)
    char_mask = lasagne.layers.ReshapeLayer(char_mask, (-1, [2]))

    char_forward = lasagne.layers.LSTMLayer(char_embed, char_hidden_size, mask_input = char_mask, nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    char_forward = lasagne.layers.ReshapeLayer(char_forward, (basize, -1, [1]))
    #char_forward = lasagne.layers.SliceLayer(char_forward, -1, axis = 2)

    char_backward = lasagne.layers.LSTMLayer(char_embed, char_hidden_size, mask_input = char_mask, nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    char_backward = lasagne.layers.ReshapeLayer(char_backward, (basize, -1, [1]))
    #char_backward = lasagne.layers.SliceLayer(char_backward, 0, axis = 2)

    char_concat = lasagne.layers.ConcatLayer([char_forward, char_backward], axis = 2)

    word_in = lasagne.layers.InputLayer(shape=(None, None), input_var = word_input_var)
    word_embed = lasagne.layers.EmbeddingLayer(word_in, len(word_dict), word_embed_size, W = word_vector)
    if not finetune:
        word_embed.params[word_embed.W].remove('trainable')

    word_mask = lasagne.layers.InputLayer(shape=(None, None), input_var = mask_var)

    word_concat = lasagne.layers.ConcatLayer([char_concat, word_embed], axis = 2)

    if embed_dropout > 0.0:
        word_concat = lasagne.layers.DropoutLayer(word_concat, p=embed_dropout)

    word_forward = lasagne.layers.LSTMLayer(word_concat, word_hidden_size, mask_input = word_mask, grad_clipping=clip, nonlinearity=lasagne.nonlinearities.tanh)
    word_backward = lasagne.layers.LSTMLayer(word_concat, word_hidden_size, mask_input = word_mask, grad_clipping=clip, nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    word_c = lasagne.layers.ConcatLayer([word_forward, word_backward], axis = 2)

    if word_dropout:
        word_c = lasagne.layers.DropoutLayer(word_c, p=word_dropout)
    #word_c = word_concat

    if self_att:
        matrix = self_att_layer(word_c, 2*word_hidden_size, mask_input=word_mask)
        #mtx = lasagne.layers.get_output(matrix)
        if att_sm_dropout > 0.0:
            matrix = lasagne.layers.DropoutLayer(matrix, p=att_sm_dropout)
        context_v = dotlayer(matrix, word_c, mask_input=word_mask)
        if att_dropout > 0.0:
            context_v = lasagne.layers.DropoutLayer(context_v, p=att_dropout)
        word_c = lasagne.layers.ConcatLayer([word_c, context_v], axis = 2)

        # word_c = self_att_layer(word_c, 2*word_hidden_size, mask_input=word_mask)
        # if att_dropout:
        #     word_c = lasagne.layers.DropoutLayer(word_c, p=0.5)

    #word_c = self_att_layer(word_c, 2*word_hidden_size, mask_input=word_mask)
    #word_c = lasagne.layers.DropoutLayer(word_c, p=0.5)

    if use_crf:

        crf = CRFLayer(word_c, len(LABEL_INDEX), mask_input = word_mask)

        energies_train = lasagne.layers.get_output(crf)
        loss_train = crf_loss(energies_train, target_var, mask_var).mean()
        train_pred, corr_train = crf_accuracy(energies_train, target_var)
        corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)

        energies_eval = lasagne.layers.get_output(crf, deterministic=True)
        loss_eval = crf_loss(energies_eval, target_var, mask_var).mean()
        prediction_eval, corr_eval = crf_accuracy(energies_eval, target_var)
        corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

        all_params = lasagne.layers.get_all_params(crf, trainable = True)
        updates = lasagne.updates.momentum(loss_train, all_params, learning_rate = lr)

        train = theano.function([char_input_var, char_mask_var, word_input_var, mask_var, target_var], [loss_train, corr_train, num_tokens], updates = updates, on_unused_input='warn')

        test_eval = theano.function([char_input_var, char_mask_var, word_input_var, mask_var, target_var], [loss_eval, corr_eval, prediction_eval, num_tokens], on_unused_input='warn')

        dimension = theano.function([word_input_var], [basize, seq_len], on_unused_input='warn')

    else:

        l_out = lasagne.layers.ReshapeLayer(word_c, (-1, [2]))
        l_out = lasagne.layers.DenseLayer(l_out, len(LABEL_INDEX), nonlinearity=lasagne.nonlinearities.softmax)

        prediction_train = lasagne.layers.get_output(l_out)
        loss_train = T.dot(lasagne.objectives.categorical_crossentropy(prediction_train, target_var.flatten()), mask_var.flatten()) / num_tokens
        corr_train = T.dot(lasagne.objectives.categorical_accuracy(prediction_train,  target_var.flatten()), mask_var.flatten())

        prediction_eval = lasagne.layers.get_output(l_out, deterministic=True)
        final_prediction = T.argmax(prediction_eval, axis=1).reshape((basize, seq_len))
        loss_eval = T.dot(lasagne.objectives.categorical_crossentropy(prediction_eval, target_var.flatten()), mask_var.flatten()) / num_tokens
        corr_eval = T.dot(lasagne.objectives.categorical_accuracy(prediction_eval, target_var.flatten()), mask_var.flatten())

        all_params = lasagne.layers.get_all_params(l_out, trainable = True)
        updates = lasagne.updates.adagrad(loss_train, all_params, learning_rate = lr)

        train = theano.function([char_input_var, char_mask_var, word_input_var, mask_var, target_var], [loss_train, corr_train, num_tokens], updates = updates, on_unused_input='warn')

        test_eval = theano.function([char_input_var, char_mask_var, word_input_var, mask_var, target_var], [loss_eval, corr_eval, final_prediction, num_tokens], on_unused_input='warn')

    best_dev = 0
    best_test = 0

    print('training...')

    def evaluate(test_X, test_Y):

        test_batch = 0
        
        test_corr = 0
        test_total = 0
        test_preds = []

        for batch in iterate_minibatches(test_X, test_Y, 100):
             
            X, Y = batch
            word_input, word_input_mask, char_input, char_input_mask, label_input = to_input_variable(X, Y, word_dict, char_dict)

            batchloss, corr, pred, num = test_eval(char_input, char_input_mask, word_input, word_input_mask, label_input)
            test_corr += corr
            test_total += num
            test_batch += 1
            for i in range(len(pred)):
                test_preds.append([LABEL_INDEX[j] for j in pred[i][0:list(word_input_mask[i]).count(1)]])

        test_acc = test_corr * 100.0 / test_total
        test_p, test_r, test_f = evaluate_ner(test_preds, test_Y)

        return test_acc, test_p, test_r, test_f

    for epoch in range(1, num_epoch + 1):

        loss = 0
        train_corr = 0
        train_batches = 0
        train_total = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_X, train_Y, batch_size, shuffle=True):

            X, Y = batch
            
            word_input, word_input_mask, char_input, char_input_mask, label_input = to_input_variable(X, Y, word_dict, char_dict)
            
            batchloss, corr, num = train(char_input, char_input_mask, word_input, word_input_mask, label_input)

            loss += batchloss
            train_corr += corr
            train_total += num
            train_batches += 1

            if log_interval > 0 and train_batches % log_interval == 0:
                d_acc, d_p, d_r, d_f = evaluate(dev_X, dev_Y)
                t_acc, t_p, t_r, t_f = evaluate(test_X, test_Y)
                    
                if d_f > best_dev:
                    best_dev = d_f
                    best_test = t_f
                    print("{:4d}/{:4d} batches |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(train_batches, len(train_X) // batch_size, d_acc, d_p, d_r, d_f))
                    print("{:4d}/{:4d} batches | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(train_batches, len(train_X) // batch_size, t_acc, t_p, t_r, t_f))
                else:
                    print("{:4d}/{:4d} batches |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(train_batches, len(train_X) // batch_size, d_acc, d_p, d_r, d_f))
                    print("{:4d}/{:4d} batches | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(train_batches, len(train_X) // batch_size, t_acc, t_p, t_r, t_f))

        d_acc, d_p, d_r, d_f = evaluate(dev_X, dev_Y)
        t_acc, t_p, t_r, t_f = evaluate(test_X, test_Y)

        print("Epoch {} of {} took {:.4f}s, learning rate: {:.4f}, training loss: {:.4f}, training accuracy: {:.4f}".format(epoch, num_epoch, time.time() - start_time, lr, loss/train_batches, train_corr * 100.0/train_total))

        if d_f > best_dev:
            best_dev = d_f
            best_test = t_f
            #save_model(lasagne.layers.get_all_param_values(crf), epoch)
            print(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(d_acc, d_p, d_r, d_f))
            print("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(t_acc, t_p, t_r, t_f))

        else:
            print(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(d_acc, d_p, d_r, d_f))
            print("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(t_acc, t_p, t_r, t_f))

        # output = open('test_pred' +`epoch`, 'w')
        # for i in range(len(test_preds)):
        #     for j in range(len(test_preds[i])):
        #         output.write(test_X[i][j] + ' ' + test_Y[i][j] + ' ' + test_preds[i][j] + '\n')
        #     output.write('\n')
        # output.close()

        lr = args.lr / (1.0 + epoch * decay)
        updates = lasagne.updates.momentum(loss_train, all_params, learning_rate=lr)
        train = theano.function([char_input_var, char_mask_var, word_input_var, mask_var, target_var], [loss_train, corr_train, num_tokens], updates = updates, on_unused_input='warn')

    print("Best test F1: " + str(best_test))

    if len(args.log_path) > 0:
        with open(args.log_path, 'a') as myfile:
            myfile.write(str(best_test) + '\n')

if __name__ == '__main__':
    main()
