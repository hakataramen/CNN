from gensim.models import word2vec
import pickle
import random
import numpy as np

import param

p_tr_path = 'Pickle/train'
p_dev_path = 'Pickle/dev'

#model_path = '../W2V/w2v_500_structuredskipgram.bin'
model_path = '../W2V/w2v_500_200_skipgram.bin'
#model_path = '../W2V/w2v_300_200_skipgram.bin'


def load_pickle(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)


# Make Dictionary
def make_dict(sent_list):
    model = word2vec.Word2Vec.load_word2vec_format(model_path, binary = True)

    dict = {}
    W_e = []

    # Make w2v dictionary
    count = 0
    for i in model.vocab.keys():
        W_e.append(model[i])
        count += 1
        dict[i] = count

    # DRUG1 DRUG2
    dict['DRUG1'] = count+1
    dict['DRUG2'] = count+2
    dict['DRUGOTHER'] = count+3
    W_e.append(model['drug'])
    W_e.append(model['drug'])
    W_e.append(model['drug'])

    index = []
    uk_num = len(dict)
    first = len(dict)
    
    hindo = {}
    # Hindo
    for i in sent_list:
        for j in i:
            try:
                hindo[j.lower()] += 1
            except:
                hindo[j.lower()] = 1
    #for key,value in hindo.items():
    #    if value <= 2:
    #        print(key)
   
    for i in sent_list:
        for j in i:
            try:
                check = dict[j.lower()]
            except:
                uk_num += 1
                dict[j.lower()] = uk_num
    vocab = len(dict)
    # Append train data dictionary
    for i in sent_list:
        line = []
        for j in i:
            #line.append(dict[j.lower()])
            if hindo[j.lower()] <= 1:
                line.append(vocab+1)
            else:
                line.append(dict[j.lower()])
        index.append(line)

    # Append train data W_e
    for i in range(uk_num - first + 1):
        ran_list = []
        for j in range(param.EMBEDDING_SIZE):
            ran = random.uniform(-param.EMBEDDING_RANGE, param.EMBEDDING_RANGE)
            #ran = random.uniform(-1, 1)
            ran_list.append(ran)
        W_e.append(ran_list)

    print('train UNK', uk_num-first)
    train_new =  uk_num-first

    return dict, W_e, train_new

# Get vocabulay index
def get_index(dic, sent_list):
    index = []
    vocab = len(dic)
    dev_unk = {}

    for i in sent_list:
        line = []
        for j in i:
            try:
                line.append(dic[j.lower()])
            except:
                line.append(vocab+1)
                dev_unk[j.lower()] = 0
        index.append(line)
    print('dev UNK', len(dev_unk))
    return index

# Get max length of senentence (train data)
def get_maxlen(sent_list):
    max_len = 0
    for i in sent_list:
        if len(i) > max_len:
            max_len = len(i)

    return max_len

# Padding
def padding(index, max_len):
    for i in index:
        if max_len - len(i) < 0:
            print('Padding Error train max length < test max length')
        for j in range(max_len - len(i)):
            i.append(0)

    return index

def word_position_index(index, pe, max_len):
    pe1_list = []
    pe2_list = []
    for i in range(len(index)):
        pe1_sent = []
        pe2_sent = []
        for j in range(len(index[i])):
            pe1_sent.append(j - pe[i][0] + max_len)
            pe2_sent.append(j - pe[i][1] + max_len)
        pe1_list.append(pe1_sent)
        pe2_list.append(pe2_sent)

    return [pe1_list, pe2_list]

def main():
    sent_tr, label_tr, pe_tr, id_tr, y_tr, y_m_tr = load_pickle(p_tr_path)
    sent_dev, label_dev, pe_dev, id_dev, y_dev, y_m_dev = load_pickle(p_dev_path)
    max_len = get_maxlen(sent_tr)
    dic, W_e, train_new = make_dict(sent_tr)
    index_tr = get_index(dic, sent_tr) 
    index_dev = get_index(dic, sent_dev) 
    wp_tr = word_position_index(index_tr, pe_tr, max_len)
    wp_dev = word_position_index(index_dev, pe_dev, max_len)
    padding(index_tr, max_len)
    padding(index_dev, max_len)
    padding(wp_tr[0], max_len)
    padding(wp_tr[1], max_len)
    padding(wp_dev[0], max_len)
    padding(wp_dev[1], max_len)

    train = [index_tr, label_tr, pe_tr, id_tr, wp_tr, y_tr, y_m_tr]
    dev = [index_dev, label_dev, pe_dev, id_dev, wp_dev, y_dev, y_m_dev]

    return [train, dev, W_e, train_new]

if __name__ == '__main__':
    main()
