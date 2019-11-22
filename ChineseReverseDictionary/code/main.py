import argparse, torch, gc, os, random, json
from data import MyDataset, load_data, my_collate_fn, device, word2sememe, label_multihot, word2POS, word2chara, word2Cn, my_collate_fn_test
from model import Encoder
from tqdm import tqdm
from evaluate import evaluate, evaluate_test
import numpy as np

def main(epoch_num, batch_size, verbose, UNSEEN, SEEN, MODE):
    [hownet_file, sememe_file, word_index_file, word_vector_file, dictionary_file, word_cilinClass_file] = ['hownet.json', 'sememe.json', 'word_index.json', 'word_vector.npy', 'dictionary_sense.json', 'word_cilinClass.json']
    word2index, index2word, word2vec, sememe_num, label_size, label_size_chara, word_defi_idx_all = load_data(hownet_file, sememe_file, word_index_file, word_vector_file, dictionary_file, word_cilinClass_file)
    (word_defi_idx_TrainDev, word_defi_idx_seen, word_defi_idx_test2000, word_defi_idx_test200, word_defi_idx_test272) = word_defi_idx_all
    index2word = np.array(index2word)
    length = len(word_defi_idx_TrainDev)
    valid_dataset = MyDataset(word_defi_idx_TrainDev[int(0.9*length):])
    test_dataset = MyDataset(word_defi_idx_test2000 + word_defi_idx_test200 + word_defi_idx_test272)
    if SEEN:
        mode = 'S_'+MODE
        print('*METHOD: Seen defi.')
        print('*TRAIN: [Train + allSeen(2000+200+272)]')
        print('*TEST: [2000rand1 + 200desc + 272desc]')
        train_dataset = MyDataset(word_defi_idx_TrainDev[:int(0.9*length)] + word_defi_idx_seen)
    elif UNSEEN:
        mode = 'U_'+MODE
        print('*METHOD: Unseen All words and defi.')
        print('*TRAIN: [Train]')
        print('*TEST: [2000rand1 + 200desc + 272desc]')
        train_dataset = MyDataset(word_defi_idx_TrainDev[:int(0.9*length)])
    print('*MODE: [%s]'%mode)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn_test)
    
    print('Train dataset: ', len(train_dataset))
    print('Valid dataset: ', len(valid_dataset))
    print('Test dataset: ', len(test_dataset))
    word_defi_idx = word_defi_idx_TrainDev + word_defi_idx_seen
        
    wd2sem = word2sememe(word_defi_idx, len(word2index), sememe_num)
    wd_sems = label_multihot(wd2sem, sememe_num)
    wd_sems = torch.from_numpy(np.array(wd_sems[:label_size])).to(device)
    wd_POSs = label_multihot(word2POS(word_defi_idx, len(word2index), 13), 13)
    wd_POSs = torch.from_numpy(np.array(wd_POSs[:label_size])).to(device)
    wd_charas = label_multihot(word2chara(word_defi_idx, len(word2index), label_size_chara), label_size_chara)
    wd_charas = torch.from_numpy(np.array(wd_charas[:label_size])).to(device)
    wd2Cilin1 = word2Cn(word_defi_idx, len(word2index), 'C1', 13)
    wd_C1 = label_multihot(wd2Cilin1, 13) #13 96 1426 4098
    wd_C1 = torch.from_numpy(np.array(wd_C1[:label_size])).to(device)
    wd_C2 = label_multihot(word2Cn(word_defi_idx, len(word2index), 'C2', 96), 96) 
    wd_C2 = torch.from_numpy(np.array(wd_C2[:label_size])).to(device)
    wd_C3 = label_multihot(word2Cn(word_defi_idx, len(word2index), 'C3', 1426), 1426) 
    wd_C3 = torch.from_numpy(np.array(wd_C3[:label_size])).to(device)
    wd_C4 = label_multihot(word2Cn(word_defi_idx, len(word2index), 'C4', 4098), 4098) 
    wd_C4 = torch.from_numpy(np.array(wd_C4[:label_size])).to(device)
    '''wd2Cilin = word2Cn(word_defi_idx, len(word2index), 'C', 5633)
    wd_C0 = label_multihot(wd2Cilin, 5633) 
    wd_C0 = torch.from_numpy(np.array(wd_C0[:label_size])).to(device)
    wd_C = [wd_C1, wd_C2, wd_C3, wd_C4, wd_C0]
    '''
    wd_C = [wd_C1, wd_C2, wd_C3, wd_C4]
    #----------mask of no sememes
    print('calculating mask of no sememes...')
    mask_s = torch.zeros(label_size, dtype=torch.float32, device=device)
    for i in range(label_size):
        sems = set(wd2sem[i].detach().cpu().numpy().tolist())-set([sememe_num])
        if len(sems)==0:
            mask_s[i] = 1

    mask_c = torch.zeros(label_size, dtype=torch.float32, device=device)
    for i in range(label_size):
        cc = set(wd2Cilin1[i].detach().cpu().numpy().tolist())-set([13])
        if len(cc)==0:
            mask_c[i] = 1
    
    model = Encoder(vocab_size=len(word2index), embed_dim=word2vec.shape[1], hidden_dim=200, layers=1, class_num=label_size, sememe_num=sememe_num, chara_num=label_size_chara)
    model.embedding.weight.data = torch.from_numpy(word2vec)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam
    best_valid_accu = 0
    DEF_UPDATE = True
    for epoch in range(epoch_num):
        print('epoch: ', epoch)
        model.train()
        train_loss = 0
        label_list = list()
        pred_list = list()
        for words_t, sememes_t, definition_words_t, POS_t, sememes, POSs, charas_t, C, C_t in tqdm(train_dataloader, disable=verbose):
            optimizer.zero_grad()
            loss, _, indices = model('train', x=definition_words_t, w=words_t, ws=wd_sems, wP=wd_POSs, wc=wd_charas, wC=wd_C, msk_s=mask_s, msk_c=mask_c, mode=MODE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            predicted = indices[:, :100].detach().cpu().numpy().tolist()
            train_loss += loss.item()
            label_list.extend(words_t.detach().cpu().numpy())
            pred_list.extend(predicted)
        train_accu_1, train_accu_10, train_accu_100 = evaluate(label_list, pred_list)
        del label_list
        del pred_list
        gc.collect()
        print('train_loss: ', train_loss/len(train_dataset))
        print('train_accu(1/10/100): %.2f %.2F %.2f'%(train_accu_1, train_accu_10, train_accu_100))
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            label_list = []
            pred_list = []
            for words_t, sememes_t, definition_words_t, POS_t, sememes, POSs, charas_t, C, C_t in tqdm(valid_dataloader, disable=verbose):
                loss, _, indices = model('train', x=definition_words_t, w=words_t, ws=wd_sems, wP=wd_POSs, wc=wd_charas, wC=wd_C, msk_s=mask_s, msk_c=mask_c, mode=MODE)
                predicted = indices[:, :100].detach().cpu().numpy().tolist()
                valid_loss += loss.item()
                label_list.extend(words_t.detach().cpu().numpy())
                pred_list.extend(predicted)
            valid_accu_1, valid_accu_10, valid_accu_100 = evaluate(label_list, pred_list)
            print('valid_loss: ', valid_loss/len(valid_dataset))
            print('valid_accu(1/10/100): %.2f %.2F %.2f'%(valid_accu_1, valid_accu_10, valid_accu_100))
            del label_list
            del pred_list
            gc.collect()
            
            if valid_accu_10>best_valid_accu:
                best_valid_accu = valid_accu_10
                print('-----best_valid_accu-----')
                #torch.save(model, 'saved.model')
                label_list = []
                pred_list = []
                for words_t, definition_words_t in tqdm(test_dataloader, disable=verbose):
                    indices = model('test', x=definition_words_t, w=words_t, ws=wd_sems, wP=wd_POSs, wc=wd_charas, wC=wd_C, msk_s=mask_s, msk_c=mask_c, mode=MODE)
                    predicted = indices[:, :1000].detach().cpu().numpy().tolist()
                    label_list.extend(words_t.detach().cpu().numpy())
                    pred_list.extend(predicted)
                test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list, pred_list)
                print('test_accu(1/10/100): %.2f %.2F %.2f %.1f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
                if epoch>10:
                    json.dump((index2word[label_list]).tolist(), open(mode+'_label_list.json', 'w'))
                    json.dump((index2word[np.array(pred_list)]).tolist(), open(mode+'_pred_list.json', 'w'))
                del label_list
                del pred_list
                gc.collect()
            
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_num', type=int, default=50)
    parser.add_argument('-v', '--verbose',default=True, action='store_false')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-b', '--batch_size', type=int, default=128) # 256
    parser.add_argument('-u', '--unseen', default=False, action='store_true')
    parser.add_argument('-s', '--seen', default=False, action='store_true')
    parser.add_argument('-m', '--mode', type=str, default='b')
    parser.add_argument('-sd', '--seed', type=int, default=543624)
    args = parser.parse_args()
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args.epoch_num, args.batch_size, args.verbose, args.unseen, args.seen, args.mode)
    