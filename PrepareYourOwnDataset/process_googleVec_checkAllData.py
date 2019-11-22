import gensim
import numpy as np
import json

fname = '../google_vec/GoogleNews-vectors-negative300.bin' 
fname_new = '../google_vec/GoogleNews-vectors-negative300.txt' 
'''
# Read google word2vec   
print('reading English word vec (bin) ...')
model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
print('saving to txt format')
model.save_word2vec_format(fname_new, binary=False)
'''
print('reading txt format vec')
EnWordVec = {}
lines = open(fname_new, encoding='utf-8').readlines()
for line in lines[1:]:
    tem = line.strip().split()
    word = tem[0]
    try:
        v = np.array([float(num) for num in tem[1:]])
    except:
        continue
    EnWordVec[word] = v                                               
print('reading finished! Num: ', len(EnWordVec)) # 2999999(drop 1st: </s>), has 'PAD' and 'PADDING' and lower case, no 'OOV' or 'oov'
'''
# Get concept target words from concept_descriptions.tok
lines = open('concept_descriptions.tok').readlines()
concept_words = []
for line in lines:
    word = line.split()[0]
    concept_words.append(word)
with open('concept_words.txt', 'w', encoding='utf-8') as f:
    for wd in concept_words:
        f.write(wd+'\n')
print('concept_words: ', len(concept_words))
'''
#target_words_file = './definitions_100000_all_head_words.txt' # we don't use this, it is not adapt to this data
vocab_words_file = './definitions_100000.vocab' # we don't use this, it is not adapt to this data
concept_words_file = './concept_words.txt'
defi_file = './definitions.tok'
concept_words = []
target_words = []
vocab_words = []
lines = open(concept_words_file, encoding='utf-8').readlines()
for line in lines:
    concept_words.append(line.strip())
#lines = open(target_words_file, encoding='utf-8').readlines()
#for line in lines:
#    target_words.append(line.strip())
lines = open(vocab_words_file).readlines()
for line in lines:
    vocab_words.append(line.strip())
    
#intersec_wrods_tar = set(list(EnWordVec)) & set(target_words)
intersec_wrods_voc = set(list(EnWordVec)) & set(vocab_words)
intersec_wrods_con = set(list(EnWordVec)) & set(concept_words)

# After wordnik_get_defi.py we get 22word-defi in concept but not in definitions.
add_to_file = './add_to_definitions.txt'
add_lines = open(add_to_file, encoding='utf-8').readlines()

f_tar = open('./target_words.txt', 'w')
lines = open(defi_file, encoding='utf-8').readlines()
vocab_wds = []
tar_wds = []
with open('./definitions.txt', 'w') as f:
    f.writelines(add_lines)
    print('added 22word-defi (in concept but not in definitions) to definitions.')
    for line in add_lines:
        tar_wd = line.split()[0]
        if tar_wd in EnWordVec:
            if tar_wd not in tar_wds:
                tar_wds.append(tar_wd)
                f_tar.write(tar_wd+'\n')
            vocab_wds.extend(line.split())
    for line in lines:
        tar_wd = line.split()[0]
        if tar_wd in EnWordVec:
            if tar_wd not in tar_wds:
                tar_wds.append(tar_wd)
                f_tar.write(tar_wd+'\n')
            vocab_wds.extend(line.split())
            f.write(line)        
f_tar.close()
print('saved target_words.txt file (have Embeddings), saved definitions.txt file (belong to those target_words have Embeddings).')
vocab_wds = set(vocab_wds)
intersec_wrods_realVoc_voc = vocab_wds & set(vocab_words)
intersec_wrods_realVoc_vec = vocab_wds & set(list(EnWordVec))
intersec_wrods_tar_vec = set(list(EnWordVec)) & set(tar_wds)
intersec_wrods_tar_con = set(concept_words) & set(tar_wds)
print('intersec_wrods (tar-vec realVoc-voc realVoc-vec con-vec con-tar): ', len(intersec_wrods_tar_vec), len(intersec_wrods_realVoc_voc), len(intersec_wrods_realVoc_vec), len(intersec_wrods_con), len(intersec_wrods_tar_con))

# Save vec (dict type)
vec_inuse = {}
for wd in tar_wds:
    vec_inuse[wd]=list(EnWordVec[wd])
for wd in intersec_wrods_realVoc_vec.difference(set(tar_wds)):
    vec_inuse[wd]=list(EnWordVec[wd])
json.dump(vec_inuse, open('./vec_inuse.json', 'w'), ensure_ascii=False, indent=4) 
# Read vec (dict type)
#vec_inuse = json.load(open('../Hill_data/vec_inuse.json'))