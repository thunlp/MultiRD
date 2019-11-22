import numpy as np
def evaluate(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    for i in range(length):
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100

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
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))

'''
def evaluate_MAP(ground_truth, prediction):
    index = 1
    correct = 0
    point = 0
    for predicted_POS in prediction:
        if predicted_POS in ground_truth:
            correct += 1
            point += (correct / index)
        index += 1
    point /= len(ground_truth)
    return point*100.

import numpy as np    
def evaluate1(ground_truth, prediction):
    length = len(ground_truth)
    ref = np.array(ground_truth)[:, np.newaxis]
    _, c = np.where(np.array(prediction)==ref)
    accu_1 = np.sum(c==0)
    accu_10 = np.sum(c<10)
    accu_100 = np.sum(c<100)
    return accu_1/length*100, accu_10/length*100, accu_100/length*100
'''