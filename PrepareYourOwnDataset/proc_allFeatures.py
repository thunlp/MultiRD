import random
import json

lines = open('wordnet_defi_500_rand1_seen.txt').readlines()
test_words_500_seen = [line.split()[0] for line in lines]

lines = open('wordnet_defi_500_rand1_unseen.txt').readlines()
test_words_500_unseen = [line.split()[0] for line in lines]

lines = open('concept_words.txt').readlines()
concept_words = [line.strip() for line in lines]

lines = open('target_words.txt').readlines()
target_words = [line.strip() for line in lines]

lines = open('definitions.txt').readlines()
word_def = {}
for line in lines:
    tar_wd = line.strip().split()[0]
    if tar_wd not in word_def:
        word_def[tar_wd] = []
    word_def[tar_wd].append(' '.join(line.strip().split()[1:]))
    
lines = open('wordnet_defi_500_rand1_seen.txt').readlines()
word_def_500rand1_seen = {}
for line in lines:
    tar_wd = line.strip().split()[0]
    if tar_wd not in word_def_500rand1_seen:
        word_def_500rand1_seen[tar_wd] = []
    word_def_500rand1_seen[tar_wd].append(' '.join(line.strip().split()[1:]))
 
lines = open('wordnet_defi_500_rand1_unseen.txt').readlines()
word_def_500rand1_unseen = {}
for line in lines:
    tar_wd = line.strip().split()[0]
    if tar_wd not in word_def_500rand1_unseen:
        word_def_500rand1_unseen[tar_wd] = []
    word_def_500rand1_unseen[tar_wd].append(' '.join(line.strip().split()[1:]))
'''
lines = open('wordnet_defi_500_all_seen.txt').readlines()
word_def_500all_seen = {}
for line in lines:
    tar_wd = line.strip().split()[0]
    if tar_wd not in word_def_500all_seen:
        word_def_500all_seen[tar_wd] = []
    word_def_500all_seen[tar_wd].append(' '.join(line.strip().split()[1:]))
'''
lines = open('word_pos_lexname.txt').readlines()
word_pos_lexname = {}
wordnet = {}
lexname_all = []
for line in lines:
    lexname = ((line.strip().split()[1]).split('|')[1]).split(',')
    wordnet[line.strip().split()[0]] = lexname
for wd in target_words:
    if wd in wordnet:
        word_pos_lexname[wd] = wordnet[wd]
        lexname_all.extend(wordnet[wd]) # lexname_all: only in use
    else:
        word_pos_lexname[wd] = []
lexname_all = set(lexname_all)
print('lexname_all in use: ', len(lexname_all)) # 45
with open('lexname_all.txt', 'w') as f:
    for ln in lexname_all:
        f.write(ln+'\n')

lines = open('root_affix_freq.txt').readlines()
rootaffix_freq = {}
for line in lines:
    rootaffix_freq[line.strip().split()[0]] = int(line.strip().split()[1])

lines = open('word_root_affix.txt').readlines()
word_root_affix = {}
rootaffix_all = []
i = 0
for line in lines:
    root_affix = line.strip().split()
    tmp = root_affix.copy()
    for r_a in tmp:
        if rootaffix_freq[r_a] < 5:
            root_affix.remove(r_a)
    word_root_affix[target_words[i]] = root_affix
    rootaffix_all.extend(root_affix)
    i += 1
rootaffix_all = set(rootaffix_all)
print('rootaffix_all in use: ', len(rootaffix_all)) # 2404
with open('rootaffix_all.txt', 'w') as f:
    for ra in rootaffix_all:
        f.write(ra+'\n')

lines = open('word_sememe.txt').readlines()
hownet = {}
word_sememes = {}
sememes_all = []
for line in lines:
    hownet[line.strip().split('||')[0]] = (line.strip().split('||')[1]).split('|')[:-1]
for wd in target_words:
    if wd in hownet:
        word_sememes[wd] = hownet[wd]
        sememes_all.extend(hownet[wd])
    else:
        word_sememes[wd] = []
sememes_all = set(sememes_all)
print('sememes_all in use: ', len(sememes_all)) # 1378
with open('sememes_all.txt', 'w') as f:
    for s in sememes_all:
        f.write(s+'\n')   
        
#======================================================
word_features = []
random.shuffle(target_words)

for wd in concept_words:
    for defi in word_def[wd]:
        feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': defi}
        word_features.append(feature)
json.dump(word_features, open('./data_defi_c.json', 'w'), ensure_ascii=False, indent=4)
# data_conc_c = json.load(open('../Hill_data/data_conc_c.json'))

word_features = []
for wd in test_words_500_seen:
    for defi in word_def_500rand1_seen[wd]:
        feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': defi}
        word_features.append(feature)
json.dump(word_features, open('./data_test_500_rand1_seen.json', 'w'), ensure_ascii=False, indent=4)    

word_features = []
for wd in test_words_500_unseen:
    for defi in word_def_500rand1_unseen[wd]:
        feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': defi}
        word_features.append(feature)
json.dump(word_features, open('./data_test_500_rand1_unseen.json', 'w'), ensure_ascii=False, indent=4)  

other_target_words = list((set(target_words).difference(set(test_words_500_unseen))))
length = len(other_target_words)

word_features = []
for wd in other_target_words[:int(length*0.9)]:
    for defi in word_def[wd]:
        feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': defi}
        word_features.append(feature)
json.dump(word_features, open('./data_train.json', 'w'), ensure_ascii=False, indent=4)    

word_features = []
for wd in other_target_words[int(length*0.9):]:
    for defi in word_def[wd]:
        feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': defi}
        word_features.append(feature)
json.dump(word_features, open('./data_dev.json', 'w'), ensure_ascii=False, indent=4) 

word_features = []
lines = open('concept_descriptions.tok').readlines()
for line in lines:
    wd = line.split()[0]
    desc = ' '.join(line.strip().split()[1:])
    feature = {'word': wd, 'lexnames': word_pos_lexname[wd], 'root_affix': word_root_affix[wd], 'sememes': word_sememes[wd], 'definitions': desc}
    word_features.append(feature)
json.dump(word_features, open('./data_desc_c.json', 'w'), ensure_ascii=False, indent=4) 