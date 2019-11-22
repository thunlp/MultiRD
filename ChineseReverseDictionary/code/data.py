import os, gc
import torch
import torch.utils.data
import numpy as np
import json
data_path = os.path.join('..', 'data')
device = torch.device('cuda')
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

class MyDataset(torch.utils.data.Dataset): 
    def __init__(self, instances):
        self.instances = instances
    
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, index):
        return self.instances[index]
        
def load_data(hownet_file, sememe_file, word_index_file, word_vector_file, dictionary_file, word_cilinClass_file):
    # all data reading
    hownet = json.load(open(os.path.join(data_path, hownet_file)))
    sememe_list = json.load(open(os.path.join(data_path, sememe_file)))
    word_index = json.load(open(os.path.join(data_path, word_index_file)))
    #index_word = json.load(open(os.path.join(data_path, index_word_file)))
    word_vector = np.load(os.path.join(data_path, word_vector_file))
    dictionary = json.load(open(os.path.join(data_path, dictionary_file)))
    word_cilin = json.load(open(os.path.join(data_path, word_cilinClass_file)))
    
    index2POS = ['无', '副', '数', '连', '助', '动', '形', '代', '拟声', '量', '名', '叹', '介'] # 12+1
    POS2index = {'无': 0, '副': 1, '数': 2, '连': 3, '助': 4, '动': 5, '形': 6, '代': 7, '拟声': 8, '量': 9, '名': 10, '叹': 11,'介': 12}

    # data in use processing
    ## word_defi
    word_defi = list()
    for word, value in dictionary.items():
        if word not in word_index: # word must in wordvec
            continue
        sememe_set = set()
        if word in hownet: # word need not in hownet
            for sense in hownet[word]:
                for sememe in sense:
                    if sememe in sememe_list:
                        sememe_set.add(sememe)
        if not len(value['POS']):
            value['POS'] = ['无']
        word_defi.append({'word': word, 'sememes': list(sememe_set), 'definition_words': value['definition_words'], 'POS': value['POS']})

    print('target words in dictionary we use: ', len(word_defi))
    ## word_set
    word_set_target = set() # target words
    word_set_defwds = set() # words in definition
    chara_set_target = set()
    for instance in word_defi:
        word_set_target.add(instance['word'])
        chara_set_target = chara_set_target.union(set([ch for ch in instance['word']]))
    #label_size = len(word_set_target) + 2 # label --> target words
    for word, value in dictionary.items():
        definition_words = value['definition_words']
        for sense in definition_words:
            for def_word in sense:
                if def_word in word_index: # we use all words with wordvec
                    word_set_defwds.add(def_word)
    word2index = dict()
    index2word = list()
    word2index['<PAD>'] = 0
    word2index['<OOV>'] = 1
    index2word.extend(['<PAD>', '<OOV>'])
    vocab_size = len(set(list(chara_set_target)+list(word_set_target)+list(word_set_defwds)))+2
    print('Vocab size (include characters & <PAD> & <OOV>): ', vocab_size)
    ## word_vec
    word2vec = np.zeros((vocab_size, word_vector.shape[1]), dtype=np.float32)
    # add all target word's characters to word2index
    cc = 0
    for ch in chara_set_target:
        index = len(word2index)
        word2index[ch] = index
        index2word.append(ch)
        if ch in word_index:
            word2vec[index, :] = word_vector[word_index[ch]]
        else:
            cc += 1
            continue # default embedding: all 0
    #print('characters not in wordvec: ', cc)
    label_size_chara = index+1 # target words' characters count
    print('characters in target words: ', label_size_chara)
    for word in word_set_target:
        if word in word2index:
            continue
        index = len(word2index)
        word2index[word] = index
        index2word.append(word)
        word2vec[index, :] = word_vector[word_index[word]]
    label_size = index+1 # all characters and target words count
    print('all characters and target words: ', label_size)
    for word in word_set_defwds:
        if word in word2index:
            continue
        index = len(word2index)
        word2index[word] = index
        index2word.append(word)
        word2vec[index, :] = word_vector[word_index[word]]
    print('vocab size: ', index+1)

    ## sememe_id
    sememe2index = dict()
    index2sememe = list()
    for sememe in sememe_list:
        sememe2index[sememe] = len(sememe2index)
        index2sememe.append(sememe)
    #----------cilin class
    C1toindex = dict()
    C2toindex = dict()
    C3toindex = dict()
    C4toindex = dict()
    index2C1 = json.load(open(os.path.join(data_path, 'C1.json')))
    index2C2 = json.load(open(os.path.join(data_path, 'C2.json')))
    index2C3 = json.load(open(os.path.join(data_path, 'C3.json')))
    index2C4 = json.load(open(os.path.join(data_path, 'C4.json')))
    for c in index2C1:
        C1toindex[c] = len(C1toindex)
    for c in index2C2:
        C2toindex[c] = len(C2toindex)
    for c in index2C3:
        C3toindex[c] = len(C3toindex)
    for c in index2C4:
        C4toindex[c] = len(C4toindex)
    #----------
    # reading description_file
    description_file1 = 'description_byHand.json' 
    description_file2 = 'description_idio_locu.json'
    word_desc_test1 = dict()
    word_desc_test2 = dict()
    description = json.load(open(os.path.join(data_path, description_file1)))
    for word, value in description.items():
        def_word_idx = list()
        for def_word in value['definition_words']:
            if def_word in word2index and def_word!=instance['word']:
                def_word_idx.append(word2index[def_word])
            else:
                def_word_idx.append(word2index['<OOV>'])
        word_desc_test1[word] = def_word_idx
    description = json.load(open(os.path.join(data_path, description_file2)))
    for word, value in description.items():
        def_word_idx = list()
        for def_word in value['definition_words']:
            if def_word in word2index and def_word!=instance['word']:
                def_word_idx.append(word2index[def_word])
            else:
                def_word_idx.append(word2index['<OOV>'])
        word_desc_test2[word] = def_word_idx
    
    ## word_def_id
    word_defi_idx_seen = list() # all defi of (2000 + 200 + 272)
    word_defi_idx_test2000 = list() # one difi of 2000
    word_defi_idx_test200 = list()  # desc of 200
    word_defi_idx_test272 = list()  # desc of 272
    word_defi_idx_TrainDev = list()
    defword_size = 0
    test2000_words = []
    for instance in word_defi:
        sememe_idx = [sememe2index[sememe] for sememe in instance['sememes']]
        POS_idx = [POS2index[pos] for pos in instance['POS']]
        chara_idx = [word2index[ch] for ch in instance['word']]
        try:
            value = word_cilin[instance['word']]
            C1_idx = [C1toindex[c] for c in value[1]] # value[0] is the whole labels --> C, then C1, C2, C3, C4
            C2_idx = [C2toindex[c] for c in value[2]]
            C3_idx = [C3toindex[c] for c in value[3]]
            C4_idx = [C4toindex[c] for c in value[4]]
        except:
            [C1_idx, C2_idx, C3_idx, C4_idx] = [[], [], [], []]
        SENSE1st_FLAG = True
        SENSE1st_FLAG200 = True
        SENSE1st_FLAG272 = True
        for sense in instance['definition_words']:
            if len(sense)<2:
                continue
            defword_size += 1
            def_word_idx = list()
            for def_word in sense:
                if def_word in word2index and def_word!=instance['word']:
                    def_word_idx.append(word2index[def_word])
                else:
                    def_word_idx.append(word2index['<OOV>'])
            
            if instance['word'] in word_desc_test1:
                word_defi_idx_seen.append({'word': word2index[instance['word']], 'chara': chara_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx, 'POS': POS_idx, 'C1': C1_idx, 'C2': C2_idx, 'C3': C3_idx, 'C4': C4_idx})
                if SENSE1st_FLAG200:
                    word_defi_idx_test200.append({'word': word2index[instance['word']], 'definition_words': word_desc_test1[instance['word']]})
                    SENSE1st_FLAG200 = False
            if instance['word'] in word_desc_test2:
                word_defi_idx_seen.append({'word': word2index[instance['word']], 'chara': chara_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx, 'POS': POS_idx, 'C1': C1_idx, 'C2': C2_idx, 'C3': C3_idx, 'C4': C4_idx})
                if SENSE1st_FLAG272:
                    word_defi_idx_test272.append({'word': word2index[instance['word']], 'definition_words': word_desc_test2[instance['word']]})
                    SENSE1st_FLAG272 = False
                    
            if len(set(test2000_words)) < 2000:
                test2000_words.append(instance['word'])
                word_defi_idx_seen.append({'word': word2index[instance['word']], 'chara': chara_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx, 'POS': POS_idx, 'C1': C1_idx, 'C2': C2_idx, 'C3': C3_idx, 'C4': C4_idx})
                if SENSE1st_FLAG: # only get the first sense if multi-sense
                    word_defi_idx_test2000.append({'word': word2index[instance['word']], 'definition_words': def_word_idx})
                    SENSE1st_FLAG = False
            elif (instance['word'] not in word_desc_test1) and (instance['word'] not in word_desc_test2):
                word_defi_idx_TrainDev.append({'word': word2index[instance['word']], 'chara': chara_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx, 'POS': POS_idx, 'C1': C1_idx, 'C2': C2_idx, 'C3': C3_idx, 'C4': C4_idx})

    print('word-defi pair size: ', defword_size)
    sememe_num = len(sememe2index)
    return word2index, index2word, word2vec, sememe_num, label_size, label_size_chara, (word_defi_idx_TrainDev, word_defi_idx_seen, word_defi_idx_test2000, word_defi_idx_test200, word_defi_idx_test272)
    
def build_sentence_numpy(sentences):
    max_length = max([len(sentence) for sentence in sentences])
    sentence_numpy = np.zeros((len(sentences), max_length), dtype=np.int64)
    for i in range(len(sentences)):
        sentence_numpy[i, 0:len(sentences[i])] = np.array(sentences[i])
    return sentence_numpy  
    
def label_multihot(labels, num):
    sm = np.zeros((len(labels), num), dtype=np.float32)
    for i in range(len(labels)):
        for s in labels[i]:
            if s >= num:
                break
            sm[i, s] = 1
    return sm
    
def my_collate_fn(batch):
    words = [instance['word'] for instance in batch]
    sememes = [instance['sememes'] for instance in batch]
    definition_words = [instance['definition_words'] for instance in batch]
    POSs = [instance['POS'] for instance in batch]
    charas = [instance['chara'] for instance in batch]
    words_t = torch.tensor(np.array(words), dtype=torch.int64, device=device)
    sememes_t = torch.tensor(label_multihot(sememes, 1400), dtype=torch.float32, device=device) # used for calculate loss of sememe prediction
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    POS_t = torch.tensor(label_multihot(POSs, 13), dtype=torch.float32, device=device) # used for calculate loss of POS prediction
    charas_t = torch.tensor(build_sentence_numpy(charas), dtype=torch.int64, device=device)
    C1 = [instance['C1'] for instance in batch]
    C2 = [instance['C2'] for instance in batch]
    C3 = [instance['C3'] for instance in batch]
    C4 = [instance['C4'] for instance in batch]
    C1_t = torch.tensor(label_multihot(C1, 13), dtype=torch.float32, device=device) #13 96 1426 4098
    C2_t = torch.tensor(label_multihot(C2, 96), dtype=torch.float32, device=device)
    C3_t = torch.tensor(label_multihot(C3, 1426), dtype=torch.float32, device=device)
    C4_t = torch.tensor(label_multihot(C4, 4098), dtype=torch.float32, device=device)
    C = [C1, C2, C3, C4]
    C_t = [C1_t, C2_t, C3_t, C4_t]
    return words_t, sememes_t, definition_words_t, POS_t, sememes, POSs, charas_t, C, C_t
    
def my_collate_fn_test(batch):
    words = [instance['word'] for instance in batch]
    definition_words = [instance['definition_words'] for instance in batch]
    words_t = torch.tensor(np.array(words), dtype=torch.int64, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    return words_t, definition_words_t
    
def word2sememe(train_dataset, word_num, sememe_num):
    max_sememe_num = max([len(instance['sememes']) for instance in train_dataset])
    ret = np.zeros((word_num, max_sememe_num), dtype=np.int64)
    ret.fill(sememe_num)
    for instance in train_dataset:
        sememes = instance['sememes']
        ret[instance['word'], :len(sememes)] = np.array(sememes)
    return torch.tensor(ret, dtype=torch.int64, device=device)
    
def word2POS(train_dataset, word_num, POS_num):
    max_POS_num = max([len(instance['POS']) for instance in train_dataset])
    ret = np.zeros((word_num, max_POS_num), dtype=np.int64)
    ret.fill(POS_num)
    for instance in train_dataset:
        POS = instance['POS']
        ret[instance['word'], :len(POS)] = np.array(POS)
    return torch.tensor(ret, dtype=torch.int64, device=device)
    
def word2chara(train_dataset, word_num, chara_num):
    max_chara_num = max([len(instance['chara']) for instance in train_dataset])
    ret = np.zeros((word_num, max_chara_num), dtype=np.int64)
    ret.fill(chara_num)
    for instance in train_dataset:
        chara = instance['chara']
        ret[instance['word'], :len(chara)] = np.array(chara)
    return torch.tensor(ret, dtype=torch.int64, device=device)
    
    
def word2Cn(train_dataset, word_num, Cn, C_num):
    max_C_num = max([len(instance[Cn]) for instance in train_dataset])
    ret = np.zeros((word_num, C_num), dtype=np.int64)
    ret.fill(C_num)
    for instance in train_dataset:
        C = instance[Cn]
        ret[instance['word'], :len(C)] = np.array(C)
    return torch.tensor(ret, dtype=torch.int64, device=device)