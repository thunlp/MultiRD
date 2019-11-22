path = './'                                                       
w = 'analyse_word_list.json'
s = 'analyse_word_syn.json'
    
import argparse
import json, os
import numpy as np

word = json.load(open(path+w))
synset = json.load(open(path+s))
lines = open('../data/target_words.txt').readlines()
target_words = [line.strip() for line in lines]
assert len(word) == 1200
def evaluate_test(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    pred_rank = []
    for i in range(length):
        try:
            pred_rank.append(prediction[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1
    return pred_rank, accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))
    
def evaluate_synset(ground_truth, prediction): # one batch
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth) # batch size
    pred_rank = []    
    for i in range(length):
        if prediction[i][0] in ground_truth[i]:
            accu_1 += 1
            accu_10 += 1
            accu_100 += 1
            pred_rank.append(0)
        elif set(prediction[i][:10]).intersection(set(ground_truth[i])):
            accu_10 += 1
            accu_100 += 1
            for j in range(10):
                if prediction[i][j] in ground_truth[i]:
                    pred_rank.append(j)
                    break
        elif set(prediction[i][:100]).intersection(set(ground_truth[i])):
            accu_100 += 1
            for j in range(100):
                if prediction[i][j] in ground_truth[i]:
                    pred_rank.append(j)
                    break
        else:
            for j in range(1000):
                if prediction[i][j] in ground_truth[i]:
                    pred_rank.append(j)
                    break
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))
    
def evaluate_1stChar(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    prediction_char = [[]]*length
    i = 0
    for gt in ground_truth:
        char1st = gt[0]
        prediction_char[i] = []
        for wd in prediction[i]:
            if wd[0] == char1st:
                prediction_char[i].append(wd)
        i += 1
    pred_rank = []
    for i in range(length):
        try:    
            pred_rank.append(prediction_char[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction_char[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction_char[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction_char[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))

def evaluate_len(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    prediction_len = [[]]*length
    i = 0
    for gt in ground_truth:
        leng = len(gt)
        prediction_len[i] = []
        for wd in prediction[i]:
            if len(wd) == leng:
                prediction_len[i].append(wd)
        i += 1
    pred_rank = []
    for i in range(length):
        try:    
            pred_rank.append(prediction_len[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction_len[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction_len[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction_len[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))

def evaluate_POS(ground_truth, prediction, word_pos):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    prediction_pos = [[]]*length
    i = 0
    for gt in ground_truth:
        pos = set(word_pos[gt])
        prediction_pos[i] = []
        for wd in prediction[i]:
            if (set(word_pos[wd]) & pos):
                prediction_pos[i].append(wd)
        i += 1
    pred_rank = []
    for i in range(length):
        try:    
            pred_rank.append(prediction_pos[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction_pos[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction_pos[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction_pos[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))


def main(mode):
    label_list_wd = json.load(open('../'+mode+'_label_list.json'))
    print('load file : '+mode+'_label_list.json'+' [OK]')
    pred_list_wd = json.load(open('../'+mode+'_pred_list.json'))
    print('load file : '+mode+'_pred_list.json'+' [OK]')

    ####### seen
    print('Test on 500: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[:500], pred_list_wd[:500])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 500 synset: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_synset(synset[:500], pred_list_wd[:500])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 500 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[:500], pred_list_wd[:500])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 500 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[:500], pred_list_wd[:500])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    if os.path.exists(path+'word_pos.json'):
        word_pos = json.load(open(path+'word_pos.json'))
    else:
        lines = open('../../Hill_data_0903/word_pos_lexname.txt').readlines()
        word_pos = {}
        word_pos['<OOV>'] = [] # this can happen sometime 
        wordnet = {}
        for line in lines:
            pos = ((line.strip().split()[1]).split('|')[0]).split(',')
            wordnet[line.strip().split()[0]] = pos
        for wd in target_words:
            if wd in wordnet:
                word_pos[wd] = wordnet[wd]
            else:
                word_pos[wd] = ['n','v','r','s','a'] # if no pos, give it all
        json.dump(word_pos, open(path+'word_pos.json', 'w'))    
    print('Test on 500 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[:500], pred_list_wd[:500], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    
    
    ####### unseen
    print('Test on 500: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[500:1000], pred_list_wd[500:1000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 500 synset: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_synset(synset[500:1000], pred_list_wd[500:1000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 500 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[500:1000], pred_list_wd[500:1000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 500 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[500:1000], pred_list_wd[500:1000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 500 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[500:1000], pred_list_wd[500:1000], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    ###### concept
    print('Test on 200: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[1000::], pred_list_wd[1000::])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 200 synset: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_synset(synset[1000::], pred_list_wd[1000::])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 200 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[1000::], pred_list_wd[1000::])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 200 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[1000::], pred_list_wd[1000::])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 200 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[1000::], pred_list_wd[1000::], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    '''
    rank_freq = [[],[],[],[],[],[]]     # freq
    rank_defNum = [[],[],[],[],[],[]]   # sen_num
    rank_defLen = [[],[],[],[],[],[]]   # defi_len
    for i in range(len(pred_rank_list)):
        if freq[i]>=30000:
            rank_freq[5].append(pred_rank_list[i])
        elif freq[i]<5000:
            rank_freq[0].append(pred_rank_list[i])
        elif freq[i]<10000:
            rank_freq[1].append(pred_rank_list[i])
        elif freq[i]<15000:
            rank_freq[2].append(pred_rank_list[i])
        elif freq[i]<20000:
            rank_freq[3].append(pred_rank_list[i])
        else:
            rank_freq[4].append(pred_rank_list[i])
                       
        if sen_num[i]>5:
            rank_defNum[5].append(pred_rank_list[i])
        else:
            rank_defNum[sen_num[i]-1].append(pred_rank_list[i])
            
        if defi_len[i]>25:
            rank_defLen[5].append(pred_rank_list[i])
        else:
            rank_defLen[int((defi_len[i]-0.1)/5)].append(pred_rank_list[i])
    json.dump(rank_freq, open(path+'rank_freq_'+mode+'.json', 'w'))  
    json.dump(rank_defLen, open(path+'rank_defLen_'+mode+'.json', 'w'))
    json.dump(rank_defNum, open(path+'rank_defNum_'+mode+'.json', 'w'))
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='[mode]')
    args = parser.parse_args()
    main(args.mode)