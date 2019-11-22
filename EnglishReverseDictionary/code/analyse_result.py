import os, json
import random
import nltk
nltk.download("wordnet") # Downloading the wordnet corpus
from nltk.corpus import wordnet as wn

data_path = '../data/'

data_test_500_rand1_seen = json.load(open(os.path.join(data_path, 'data_test_500_rand1_seen.json')))
data_test_500_rand1_unseen = json.load(open(os.path.join(data_path, 'data_test_500_rand1_unseen.json')))
data_desc_c = json.load(open(os.path.join(data_path, 'data_desc_c.json')))

word_list = []
for data in data_test_500_rand1_seen:
    word_list.append(data['word'])              # word
for data in data_test_500_rand1_unseen:
    word_list.append(data['word'])              # word

#lines = open(os.path.join(data_path, 'concept_words.txt')).readlines()
#concept_words = [line.strip() for line in lines]
concept_words = [value['word'] for value in data_desc_c]
word_list = word_list + concept_words

all_synsets = list(wn.all_synsets())
word_synset = {}
for synset in all_synsets:
    # filter all multi-word phrases indicated by _
    lemmas = [lemma for lemma in synset.lemmas() if "_" not in lemma.name()]
    if len(lemmas) == 0:
        continue
    for lemma in lemmas:
        wd = lemma.name().lower()
        if wd in word_list: 
            if wd not in word_synset:
                word_synset[wd] = []    
            tmp = [le.name().lower() for le in lemmas]
            word_synset[wd].extend(tmp)
word_syn = []
for wd in word_list:
    try:
        word_syn.append(list(set(word_synset[wd])))
    except:
        word_syn.append([wd])
#word_syn = [list(set(word_synset[wd])) for wd in word_list]
  
json.dump(word_syn, open('analyse_word_syn.json', 'w'))
json.dump(word_list, open('analyse_word_list.json', 'w'))
