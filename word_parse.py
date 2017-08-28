import glob
import itertools
import pickle

from geniatagger import GeniaTagger
tagger = GeniaTagger('/home/asada.13003/ddi_cnn/geniatagger-3.0.2/geniatagger')

train_path = 'Divide/train/*.ann'
dev_path = 'Divide/dev/*.ann'

p_tr_path = 'Pickle/train'
p_dev_path = 'Pickle/dev'


def sent_label_pe(path):
    sent = []
    label = []
    y = []
    y_minus = []
    id = []
    for i in glob.glob(path):
        f = open(i, 'r')
        f_txt = open(i.replace('Divide', 'Brat').replace('.ann', '.txt'), 'r')

        sentID = i.split('/')[-1].replace('.ann', '')

        line = f.readlines()
        text = f_txt.read()

        entity = []
        relation = []
        for j in line:
            tab = j.split('\t')
            if list(tab[0])[0] == 'T':
                entity.append([tab[0],tab[3]])
            elif list(tab[0])[0] == 'R':
                type = j.split('\t')[1].split()[0]
                arg1 = j.split('\t')[1].split()[1].split(':')[1]
                arg2 = j.split('\t')[1].split()[2].split(':')[1]
                relation.append([type,arg1,arg2])

        # Combination
        text_row = list(text)
        for j in itertools.combinations(entity,2):
            text_list = text_row[:]
            sum_off = 0
            for k in line:
                tab = k.split('\t')
                if tab[0] == j[0][0]:
                    name = ' DRUG1 '
                elif tab[0] == j[1][0]:
                    name = ' DRUG2 '
                elif list(tab[0])[0] == 'T':
                    name = ' DRUGOTHER '
                else:
                    continue
                    
                _, off1, off2 = tab[1].split()
                off1 = int(off1)
                off2 = int(off2)
                len_name = len(list(name))
                text_list[off1-sum_off:off2-sum_off] = list(name)
                sum_off += off2 - off1 - len_name
            s = ''
            for k in text_list:
                s += k
            sent.append(s)

            # Make label list
            flag = False
            for k in relation:
                if k[1]==j[0][0] and k[2]==j[1][0]:
                    flag = True
                    if k[0]=='MECHANISM':
                        label.append([0,1,0,0,0])
                        y.append([1])
                        y_minus.append([0,2,3,4])
                    if k[0]=='EFFECT':
                        label.append([0,0,1,0,0])
                        y.append([2])
                        y_minus.append([0,1,3,4])
                    if k[0]=='ADVISE':
                        label.append([0,0,0,1,0])
                        y.append([3])
                        y_minus.append([0,1,2,4])
                    if k[0]=='INT':
                        label.append([0,0,0,0,1])
                        y.append([4])
                        y_minus.append([0,1,2,3])
            if not flag:
                label.append([1,0,0,0,0])
                y.append([0])
                y_minus.append([1,2,3,4])
                    
            id.append([sentID, j[0][1].replace('\n', ''), j[1][1].replace('\n', '')])

        f.close()
        f_txt.close()


    sent_list = []
    for i in sent:
        word_list = []
        for parse in tagger.parse(i):
            word_list.append(parse[0])
        sent_list.append(word_list)

    # Get Target entity Position list
    pe = []
    for i in sent_list:
        try:
            pe.append([i.index('DRUG1'), i.index('DRUG2')])
        except:
            pe.append([0,0])
            print('ERROR DRUG1 or DRUG2 cant be found')
            print(i)

    return sent_list, label, pe, id, y, y_minus
        

def dump_pickle(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)
    
if __name__ == '__main__':
    sent_tr, label_tr, pe_tr, id_tr, y_tr, y_m_tr = sent_label_pe(train_path)
    sent_dev, label_dev, pe_dev, id_dev, y_dev, y_m_dev = sent_label_pe(dev_path)

    dump_pickle([sent_tr, label_tr, pe_tr, id_tr, y_tr, y_m_tr], p_tr_path)
    dump_pickle([sent_dev, label_dev, pe_dev, id_dev, y_dev, y_m_dev], p_dev_path)
