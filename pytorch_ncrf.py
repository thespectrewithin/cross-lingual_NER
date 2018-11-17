import time
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#from transformer.Models import Encoder
import constants as Constants
#from transformer.Optim import ScheduledOptim
from pytorch_crf import ChainCRF
from util import create_char_index, create_word_index, load_data, evaluate_ner, iterate_minibatches

CAT = ['PER', 'ORG', 'LOC', 'MISC']
POSITION = ['I', 'B']

LABEL_INDEX = [Constants.PAD_WORD] + ['O'] + ["{}-{}".format(position, cat) for cat in CAT for position in POSITION]


class lstmcrf(nn.Module):

    def __init__(self, wembed_size, cembed_size, whidden_size, chidden_size, n_wvocab, n_cvocab, n_label, use_crf=True, finetune=False, self_att=False, pretrained=None, emb_dropout=0.5, word_dropout=0.5, att_sm_dropout=0.0, att_dropout=0.0):
        
        super(lstmcrf, self).__init__()
        
        self.word_embed = nn.Embedding(n_wvocab, wembed_size, padding_idx=Constants.PAD)
        self.char_embed = nn.Embedding(n_cvocab, cembed_size, padding_idx=Constants.PAD)
        
        self.char_lstm = nn.LSTM(cembed_size, chidden_size, batch_first=True, bidirectional=True)
        self.word_lstm = nn.LSTM(2*chidden_size + wembed_size, whidden_size, batch_first=True, bidirectional=True, dropout=word_dropout)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.chidden_size = chidden_size

        self.use_crf = use_crf
        self.finetune = finetune
        self.self_att = self_att
        
        if self.self_att:
            self.W1 = nn.Linear(whidden_size * 2, whidden_size * 2)

            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(dim=2)

            self.att_dropout = nn.Dropout(att_dropout)
            self.att_sm_dropout = nn.Dropout(att_sm_dropout)

            if self.use_crf:
                self.crf = ChainCRF(whidden_size * 4, n_label)
            else:
                self.read_out = nn.Linear(whidden_size * 4, n_label, bias=False)
        else:
            if self.use_crf:
                self.crf = ChainCRF(whidden_size * 2, n_label)
            else:
                self.read_out = nn.Linear(whidden_size * 2, n_label, bias=False)

        if pretrained is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(pretrained))

    def forward(self, words, chars, word_length, char_length, target):

        char_input = chars.view(-1, chars.size(2))
        char_length, char_idx = char_length.view(-1).sort(0, descending=True)

        char_embedded = self.char_embed(char_input)
        char_embedded = char_embedded[char_idx]

        if char_length.data.eq(0).sum() == 0:
            zero_index = char_length.size(0)
        else:
            zero_index = char_length.data.eq(0).nonzero()[0][0]
        clen = char_length.size(0)
        char_embedded = char_embedded[:zero_index]
        char_length = char_length[:zero_index]

        char_packed_input = pack_padded_sequence(char_embedded, char_length.cpu().data.numpy(), batch_first=True)
        char_packed_output, (last_hidden, last_cell) = self.char_lstm(char_packed_input)
        char_hidden = torch.cat([last_hidden[0], last_hidden[1]], 1)
        
        if zero_index != clen:
            filler = Variable(torch.zeros((clen - zero_index, 2*self.chidden_size)), volatile=char_input.volatile, requires_grad=False).cuda()
            char_hidden = torch.cat([char_hidden, filler], 0)

        char_hidden = char_hidden[torch.from_numpy(np.argsort(char_idx.cpu().data.numpy())).cuda()]
        char_hidden = char_hidden.view(chars.size(0), -1, char_hidden.size(1))

        word_embedded = self.word_embed(words)
        word_concat = torch.cat([word_embedded, char_hidden], 2)

        if self.emb_dropout.p > 0.0:
            word_concat = self.emb_dropout(word_concat)
        
        word_length, word_idx = word_length.sort(0, descending=True)
        word_concat = word_concat[word_idx]

        word_packed_input = pack_padded_sequence(word_concat, word_length.cpu().data.numpy(), batch_first=True)
        word_packed_output, _ = self.word_lstm(word_packed_input)
        word_output, _ = pad_packed_sequence(word_packed_output, batch_first=True)
        word_output = word_output[torch.from_numpy(np.argsort(word_idx.cpu().data.numpy())).cuda()]

        if self.self_att:

            att_padding_mask = Variable(words.data.ne(Constants.PAD)).cuda()
            attention_self_mask = Variable(1 - torch.eye(words.size(1), words.size(1))).cuda()

            q = self.tanh(self.W1(word_output))
            q = q * att_padding_mask.float().unsqueeze(2)

            out = q.bmm(q.transpose(1, 2))
            out = out * attention_self_mask.unsqueeze(0)

            matrix = self.softmax(out)
            matrix = matrix * att_padding_mask.float().unsqueeze(2)
            matrix = matrix * att_padding_mask.float().unsqueeze(1)
            if self.att_sm_dropout.p > 0.0:
                matrix = self.att_sm_dropout(matrix)

            context_v = matrix.bmm(word_output)
            if self.att_dropout.p > 0.0:
                context_v = self.att_dropout(context_v)
            word_output = torch.cat([word_output, context_v], 2)

        if self.use_crf:
            crf_loss = self.crf.loss(word_output, target, words.ne(Constants.PAD).float()).mean()
            predict = self.crf.decode(word_output, words.ne(Constants.PAD).float(), 1)
            return crf_loss, predict
        else:
            return self.read_out(word_output.view(-1, word_output.size(2)))

    def get_trainable_parameters(self):
        if not self.finetune:
            freezed_param_ids = set(map(id, self.word_embed.parameters()))
            return (p for p in self.parameters() if id(p) not in freezed_param_ids)
        else:
            return self.parameters()
        
def to_input_variable(data, label, word_vocab, char_vocab, label_index, istest=False):

    word_max_seq_len = max(len(s) for s in data)
    char_max_seq_len = max(len(s) for w in data for s in w)

    word_input = np.zeros((len(data), word_max_seq_len), dtype='int64')

    char_input = np.zeros((len(data), word_max_seq_len, char_max_seq_len), dtype='int64')

    label_input = np.zeros((len(data), word_max_seq_len), dtype='int64')

    word_length_input = [len(s) for s in data]
    char_length_input = np.zeros((len(data), word_max_seq_len), dtype='int64')

    for i in range(len(data)):
        for j in range(len(data[i])):
            char_length_input[i][j] = len(data[i][j])
            if data[i][j].lower() in word_vocab:
                word_input[i][j] = word_vocab[data[i][j].lower()]
            else:
                word_input[i][j] = Constants.UNK
            label_input[i][j] = label_index.index(label[i][j])
            for k in range(len(data[i][j])):
                c = data[i][j][k]
                if c in char_vocab:
                    char_input[i][j][k] = char_vocab[c if not c.isdigit() else '0']
                else:
                    char_input[i][j][k] = Constants.UNK

#    pos = np.array([[pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(s)] for s in word_input], dtype='int64')
                
    word_var = Variable(torch.from_numpy(word_input), volatile=istest, requires_grad=False)
    label_var = Variable(torch.from_numpy(label_input), volatile=istest, requires_grad=False)
#    pos_var = Variable(torch.from_numpy(pos), volatile=istest, requires_grad=False)
    char_var = Variable(torch.from_numpy(char_input), volatile=istest, requires_grad=False)
    word_length_var = Variable(torch.LongTensor(word_length_input), volatile=istest, requires_grad=False)
    char_length_var = Variable(torch.from_numpy(char_length_input), volatile=istest, requires_grad=False)

    return word_var.cuda(), char_var.cuda(), label_var.cuda(), word_length_var.cuda(), char_length_var.cuda()

def main():

    parser = argparse.ArgumentParser(description='pytorch simple neural crf')

    #parser.add_argument('--model', type=str, choices=['transformer', 'lstm'])
    
    parser.add_argument('--wembed_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--cembed_size', type=int, default=25, help='char embedding size')
    parser.add_argument('--word_hidden', type=int, default=200, help='number of hidden units word layer')
    parser.add_argument('--char_hidden', type=int, default=50, help='number of hidden units char layer')
    parser.add_argument('--vocab_size', type=int, default=100000, help='vocab size')
    
    # parser.add_argument('--n_layer', type=int, default=2, help='number of layers')
    # parser.add_argument('--n_head', type=int, default=4, help='number of head')

    parser.add_argument('--embed_dropout', type=float, default=0.5, help='embed level dropout rate')
    parser.add_argument('--word_dropout', type=float, default=0.5, help='word level dropout rate')
    parser.add_argument('--att_sm_dropout', type=float, default=0.0, help='att softmax level dropout rate')
    parser.add_argument('--att_dropout', type=float, default=0.0, help='att level dropout rate')

    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--nepoch', type=int, default=30, help='number of training epochs')
    # parser.add_argument('--rho', type=float, default=0.95, help='rho')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='learning rate decay')
    parser.add_argument('--clip', type=float, default=5.0, help='grad clipping')
    #parser.add_argument('--warmup', type=int, default=500, help='warmup step for noam')

    parser.add_argument('--seed', type=int, default=1001, help='random seed')

    parser.add_argument('--finetune', action='store_true', help='fine tune word embedding')
    parser.add_argument('--use_crf', action='store_true',help='use crf')
    parser.add_argument('--self_att', action='store_true',help='use self attention')

    parser.add_argument('--train_data', type=str, help='training data path')
    parser.add_argument('--dev_data', type=str, help='development data path')
    parser.add_argument('--test_data', type=str, help='test data path')
    parser.add_argument('--word_vec_path', type=str, help='word vector path')

    parser.add_argument('--log_interval', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='', help='records best test results for multiple runs')

    args = parser.parse_args()

    word_embed_size = args.wembed_size
    char_embed_size = args.cembed_size
    word_hidden_size = args.word_hidden
    char_hidden_size = args.char_hidden
    vocab_size = args.vocab_size

    #n_layer = args.n_layer
    #n_head = args.n_head
    embed_dropout = args.embed_dropout
    word_dropout = args.word_dropout
    att_sm_dropout = args.att_sm_dropout
    att_dropout = args.att_dropout

    batch_size = args.batch_size
    num_epoch = args.nepoch
    lr = args.lr
    decay = args.lr_decay
    #rho = args.rho
    clip = args.clip
    #warmup = args.warmup

    use_crf = args.use_crf
    finetune = args.finetune
    self_att = args.self_att

    log_interval = args.log_interval
    
    seed = args.seed

    train_data_path = args.train_data
    dev_data_path = args.dev_data
    test_data_path = args.test_data
    word_vec_path = args.word_vec_path

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('loading data...')

    train_X, train_Y = load_data(train_data_path, 1, True)
    dev_X, dev_Y = load_data(dev_data_path, 1, True)
    test_X, test_Y = load_data(test_data_path, 1, True)

    longest = max(len(s) for s in [train_X, dev_X, test_X])

    char_dict = create_char_index([train_data_path, dev_data_path, test_data_path], True)
    word_dict, word_vector = create_word_index(word_vec_path, vocab_size, word_embed_size, True)

    #    if args.model == 'lstm':
    model = lstmcrf(word_embed_size, char_embed_size, word_hidden_size, char_hidden_size, len(word_vector), len(char_dict), len(LABEL_INDEX), use_crf, finetune, self_att, word_vector, embed_dropout, word_dropout, att_sm_dropout, att_dropout)
    #    elif args.model == 'transformer':
    #    model = Encoder(word_vector, len(word_vector), longest, len(LABEL_INDEX), len(char_dict), char_embed_size, char_hidden_size, use_crf, no_finetune, n_layers=n_layer, n_head=n_head, d_k=int(word_hidden_size/n_head), d_v=int(word_hidden_size/n_head), d_word_vec=word_embed_size, d_model=word_hidden_size, d_inner_hid=4 * word_hidden_size, dropout=dropout)
    model = model.cuda()

    vocab_mask = torch.ones(len(LABEL_INDEX))
    vocab_mask[Constants.PAD] = 0
    cross_entropy = nn.CrossEntropyLoss(weight=vocab_mask).cuda()

    optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=lr, momentum=0.9)
    #optimizer = torch.optim.Adadelta(model.get_trainable_parameters(), lr=lr, rho=rho)
    #optimizer = ScheduledOptim(
        # optim.Adam(
        #     model.get_trainable_parameters(),
        #     betas=(0.9, 0.98), eps=1e-09),
        # word_hidden_size, warmup)

    best_dev = 0
    best_test = 0

    print('training...')

    def evaluate(test_X, test_Y):

        model.eval()

        test_batch = 0

        test_corr = 0
        test_total = 0
        test_preds = []

        for batch in iterate_minibatches(test_X, test_Y, 100):

            X, Y = batch

            word_input, char_input, label_input, word_length_input, char_length_input = to_input_variable(X, Y, word_dict, char_dict, LABEL_INDEX, istest=True)

            if use_crf:
                #if args.model == 'lstm':
                loss, predict = model(word_input, char_input, word_length_input, char_length_input, label_input)
                #elif args.model == 'transformer':
                #    loss, predict = model(word_input, pos_input, char_input, char_length_input, label_input)
                pred = predict.view(-1)
            else:
                #if args.model == 'lstm':
                scores = model(word_input, char_input, word_length_input, char_length_input, label_input)
                #elif args.model == 'transformer':
                #    scores = model(word_input, pos_input, char_input, char_length_input, label_input)
                pred = scores.max(1)[1].data

            gold = label_input.contiguous().view(-1)
            num_tokens = gold.data.ne(Constants.PAD).sum()
            correct = pred.eq(gold.data).masked_select(gold.ne(Constants.PAD).data).sum()

            test_corr += correct
            test_total += num_tokens
            test_batch += 1

            pred = pred.view(len(X), -1)

            for i in range(len(pred)):
                test_preds.append([LABEL_INDEX[j] for j in pred[i][:word_length_input.data[i]]])

        test_acc = test_corr * 100.0 / test_total
        test_p, test_r, test_f = evaluate_ner(test_preds, test_Y)

        return test_acc, test_p, test_r, test_f, test_preds

    for epoch in range(1, 1+num_epoch):

        model.train()

        total_loss = 0
        total_words = 0
        total_correct = 0
        train_batches = 0
        start_time = time.time()
        
        for batch in iterate_minibatches(train_X, train_Y, batch_size, shuffle=True):

            X, Y = batch

            word_input, char_input, label_input, word_length_input, char_length_input = to_input_variable(X, Y, word_dict, char_dict, LABEL_INDEX)
            
            optimizer.zero_grad()
            gold = label_input.contiguous().view(-1)

            if use_crf:
                #if args.model == 'lstm':
                loss, predict = model(word_input, char_input, word_length_input, char_length_input, label_input)
                #elif args.model == 'transformer':
                #    loss, predict = model(word_input, pos_input, char_input, char_length_input, label_input)
                pred = predict.view(-1)
            else:
                #if args.model == 'lstm':
                scores = model(word_input, char_input, word_length_input, char_length_input, label_input)
                #elif args.model == 'transformer':
                #    scores = model(word_input, pos_input, char_input, char_length_input, label_input)
                pred = scores.max(1)[1].data
                loss = cross_entropy(scores, gold)

            num_tokens = gold.data.ne(Constants.PAD).sum()
            correct = pred.eq(gold.data).masked_select(gold.ne(Constants.PAD).data).sum()

            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm(model.get_trainable_parameters(), clip)
            optimizer.step()
            #lr = optimizer.update_learning_rate()

            total_loss += loss.data[0]
            total_correct += correct
            total_words += num_tokens
            train_batches += 1

            if log_interval > 0 and train_batches % log_interval == 0:
                d_acc, d_p, d_r, d_f, d_preds = evaluate(dev_X, dev_Y)
                t_acc, t_p, t_r, t_f, t_preds = evaluate(test_X, test_Y)
                if d_f > best_dev:
                    best_dev = d_f
                    best_test = t_f
                    print("{:4d}/{:4d} batches |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(train_batches, len(train_X) // batch_size, d_acc, d_p, d_r, d_f))
                    print("{:4d}/{:4d} batches | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(train_batches, len(train_X) // batch_size, t_acc, t_p, t_r, t_f))
                else:
                    print("{:4d}/{:4d} batches |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(train_batches, len(train_X) // batch_size, d_acc, d_p, d_r, d_f))
                    print("{:4d}/{:4d} batches | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(train_batches, len(train_X) // batch_size, t_acc, t_p, t_r, t_f))

        d_acc, d_p, d_r, d_f, d_preds = evaluate(dev_X, dev_Y)
        t_acc, t_p, t_r, t_f, t_preds = evaluate(test_X, test_Y)
        
        print("Epoch {} of {} took {:.4f}s, learning rate: {:.6f}, training loss: {:.4f}, training accuracy: {:.4f}".format(epoch, num_epoch, time.time() - start_time, lr, total_loss/train_batches, total_correct * 100.0/total_words))

        if d_f > best_dev:
            best_dev = d_f
            best_test = t_f
            print(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(d_acc, d_p, d_r, d_f))
            print("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(t_acc, t_p, t_r, t_f))

        else:
            print(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(d_acc, d_p, d_r, d_f))
            print("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(t_acc, t_p, t_r, t_f))

        lr = args.lr / (1.0 + epoch * decay)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print("Best test F1: " + str(best_test))

    if len(args.log_path) > 0:
        with open(args.log_path, 'a') as myfile:
            myfile.write(str(best_test) + '\n')

if __name__ == '__main__':
    main()
