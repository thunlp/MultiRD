import json
from evaluate import evaluate_test
import argparse

def main(mode):
    label_list_wd = json.load(open(mode+'_label_list.json'))
    print('load file : '+mode+'_label_list.json'+' [OK]')
    pred_list_wd = json.load(open(mode+'_pred_list.json'))
    print('load file : '+mode+'_pred_list.json'+' [OK]')
    if len(label_list_wd)==1200:
        print('Test on 500 seen: ')
        test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[:500], pred_list_wd[:500])
        print('test_accu(1/10/100): %.2f %.2F %.2f %.2f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
        print('Test on 500 unseen: ')
        test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[500:1000], pred_list_wd[500:1000])
        print('test_accu(1/10/100): %.2f %.2F %.2f %.2f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
        print('Test on 200: ')
        test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list_wd[1000:], pred_list_wd[1000:])
        print('test_accu(1/10/100): %.2f %.2F %.2f %.2f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='[mode]')
    args = parser.parse_args()
    main(args.mode)