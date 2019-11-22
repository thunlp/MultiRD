import random
import nltk
#nltk.download("wordnet") # Downloading the wordnet corpus
from nltk.corpus import wordnet as wn

def load_filterlist(path):
    with open(path) as input_file:
        filterlist = input_file.read().splitlines()
    return set(filterlist)
    
all_synsets = list(wn.all_synsets())
word_pos_lexname = {}
target_words = load_filterlist('./target_words.txt')   
for synset in all_synsets:
     # filter all multi-word phrases indicated by _
    lemmas = [lemma for lemma in synset.lemmas() if "_" not in lemma.name()]
    if len(lemmas) == 0:
        continue
    pos = synset.pos()
    lexname = synset.lexname()
    for lemma in lemmas:
        wd = lemma.name().lower()
        if wd in target_words: # when wd.lower() used, 41753, if not, 38113.
            if wd not in word_pos_lexname:
                word_pos_lexname[wd] = [[],[]]    
            word_pos_lexname[wd][0].append(pos)
            word_pos_lexname[wd][1].append(lexname)
with open('word_pos_lexname.txt', 'w') as f:
    for wd in word_pos_lexname:
        f.write(wd+' ')
        wpl = word_pos_lexname[wd]
        f.write(','.join(list(set(wpl[0]))))
        f.write('|')
        f.write(','.join(list(set(wpl[1]))))
        f.write('\n')
print('saved POS and lexname to file.')