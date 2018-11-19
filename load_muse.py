import numpy as np
from numpy import linalg as LA
import argparse
import torch

def nn(esp_vec, eng_vec, O, bs):

    esp = torch.from_numpy(esp_vec.astype('float32')).cuda()
    eng = torch.from_numpy(eng_vec.astype('float32')).cuda()
    O = torch.from_numpy(O.astype('float32')).cuda()

    indexes = []

    for i in range(0, len(eng_vec), bs):

        cossim = esp.mm(O).mm(eng[i:i+bs].t()).t()
        indexes.append(torch.max(cossim, 1)[1].cpu())

    return torch.cat(indexes).numpy()

def get_nn_avg_dist(esp, eng, O, k, bs):

    distance_esp = []
    distance_eng = []
    
    for i in range(0, esp.size(0), bs):
        
        cossim_esp = esp[i:i+bs].mm(O).mm(eng.t())
        esp_dist, _ = torch.topk(cossim_esp, k)
        distance_esp.append(esp_dist.mean(1))


    for i in range(0, eng.size(0), bs):

        cossim_eng = esp.mm(O).mm(eng[i:i+bs].t()).t()
        eng_dist, _ = torch.topk(cossim_eng, k)
        distance_eng.append(eng_dist.mean(1))

    return torch.cat(distance_esp), torch.cat(distance_eng)

def csls(esp_vec, eng_vec, O, k, bs):

    esp = torch.from_numpy(esp_vec.astype('float32')).cuda()
    eng = torch.from_numpy(eng_vec.astype('float32')).cuda()
    O = torch.from_numpy(O.astype('float32')).cuda()

    esp_distance, eng_distance = get_nn_avg_dist(esp, eng, O, k, bs)

    all_scores = []
    indexes = []

    for i in range(0, len(eng_vec), bs):

        cossim = esp.mm(O).mm(eng[i:i+bs].t()).t()
        scores = cossim * 2 - eng_distance[i:i+bs].unsqueeze(1) - esp_distance.unsqueeze(0)
        indexes.append(torch.max(scores, 1)[1].cpu())

    return torch.cat(indexes).numpy()

def load_embedding(path, vocab_size):
    
    word_vector = []
    word_dict = {}
    words = []
    
    num = 0

    for line in open(path):
        if num < vocab_size:
            word, vec = line.rstrip().split(' ', 1)
            word_dict[word] = len(word_dict)
            words.append(word)
            vec = np.array(vec.split(), dtype='float32')
            word_vector.append(vec)
        num+=1
    print(len(word_vector))

    return word_dict, normalize(np.vstack(word_vector)), words

def normalize(vectors):
    return vectors / LA.norm(vectors, axis=1).reshape((vectors.shape[0], 1))

def main():

    parser = argparse.ArgumentParser(description='load muse')

    parser.add_argument('--eng_path', type=str, help='eng embedding path')
    parser.add_argument('--esp_path', type=str, help='esp embedding path')
    parser.add_argument('--o_path', type=str, help='mapping path')
    parser.add_argument('--output_path', type=str, help='word-to-word translation file output path')

    parser.add_argument('--k', type=int, default=10, help='k in csls')
    parser.add_argument('--batch_size', type=int, default=5000, help='how many words to translate at once')
    parser.add_argument('--distance', type=str, default='csls', help='distance type, nn or csls')

    parser.add_argument('--vocab_size', type=int, default=100000, help='vocab size')

    args = parser.parse_args()

    eng_word, eng_vec, engs = load_embedding(args.eng_path, args.vocab_size)
    esp_word, esp_vec, esps = load_embedding(args.esp_path, args.vocab_size)

    O = torch.load(args.o_path)

    if args.distance == 'csls':
        indexes = csls(esp_vec, eng_vec, O, args.k, args.batch_size)
    elif args.distance == 'nn':
        indexes = nn(esp_vec, eng_vec, O, args.batch_size)

    output = open(args.output_path, 'w')

    for i in range(len(engs)):
        output.write(engs[i] + ' ' + esps[indexes[i]] + '\n')

    output.close()

if __name__ == '__main__':
    main()

