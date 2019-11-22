  
import argparse
import json, os
import numpy as np




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
    for i in range(length):
        if prediction[i][0] in ground_truth[i]:
            accu_1 += 1
            accu_10 += 1
            accu_100 += 1
        elif set(prediction[i][:10]).intersection(set(ground_truth[i])):
            accu_10 += 1
            accu_100 += 1
        elif set(prediction[i][:100]).intersection(set(ground_truth[i])):
            accu_100 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100
    
def evaluate_1stChar(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    prediction_char = [[]]*length
    i = 0
    for gt in ground_truth:
        if len(gt)==1: # 中文中要排除只有一个字的情况，当只有一个字时，仍然用原来的预测结果，不进行已知字的筛选。
            prediction_char[i] = prediction[i]
            i += 1
            continue
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
            try:
                if (set(word_pos[wd]) & pos):
                    prediction_pos[i].append(wd)
            except:
                prediction_pos[i].append(wd) # 为什么会有没词性的？
                #print(wd)
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
    label_list_wd = json.load(open(mode+'_label_list.json'))
    print('load file : '+mode+'_label_list.json'+' [OK]')
    pred_list_wd = json.load(open(mode+'_pred_list.json'))
    print('load file : '+mode+'_pred_list.json'+' [OK]')

    synset_all = dict()
    with open('../data/word2synset_synset.txt') as f:
        for line in f.readlines():
            wd_l = line.split()
            synset_all[wd_l[0]] = wd_l # it must include itself
    synset = []
    for wd in label_list_wd:
        if wd in synset_all:
            synset.append(synset_all[wd])
        else:
            synset.append(wd)
            
    diction = json.load(open('../data/dictionary_sense.json'))
    word_pos = {}
    word_pos['<OOV>'] = []
    for wd in diction:
        if diction[wd]['POS'] == []:
            word_pos[wd] = ['介', '副', '数', '连', '助', '动', '形', '代', '拟声', '量', '名', '叹']
        else:
            word_pos[wd] = diction[wd]['POS']
        
    print('Test on 2000: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[:2000], pred_list_wd[:2000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 2000 synset: ')
    test_accu_1, test_accu_10, test_accu_100 = evaluate_synset(synset[:2000], pred_list_wd[:2000])
    print('test_accu(1/10/100): %.2f %.2F %.2f'%(test_accu_1, test_accu_10, test_accu_100))

    print('Test on 2000 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[:2000], pred_list_wd[:2000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 2000 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[:2000], pred_list_wd[:2000])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
       
    print('Test on 2000 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[:2000], pred_list_wd[:2000], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    
    print('Test on 200: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[2000:2200], pred_list_wd[2000:2200])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 200 synset: ')
    test_accu_1, test_accu_10, test_accu_100 = evaluate_synset(synset[2000:2200], pred_list_wd[2000:2200])
    print('test_accu(1/10/100): %.2f %.2F %.2f'%(test_accu_1, test_accu_10, test_accu_100))

    print('Test on 200 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[2000:2200], pred_list_wd[2000:2200])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 200 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[2000:2200], pred_list_wd[2000:2200])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 200 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[2000:2200], pred_list_wd[2000:2200], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 272: ')
    pred_rank_list, test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[2200:], pred_list_wd[2200:])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 272 synset: ')
    test_accu_1, test_accu_10, test_accu_100 = evaluate_synset(synset[2200:], pred_list_wd[2200:])
    print('test_accu(1/10/100): %.2f %.2F %.2f'%(test_accu_1, test_accu_10, test_accu_100))

    print('Test on 272 char1st: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_1stChar(label_list_wd[2200:], pred_list_wd[2200:])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
    
    print('Test on 272 wordLen: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_len(label_list_wd[2200:], pred_list_wd[2200:])
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

    print('Test on 272 POS: ')
    test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_POS(label_list_wd[2200:], pred_list_wd[2200:], word_pos)
    print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='[mode]')
    args = parser.parse_args()
    main(args.mode)