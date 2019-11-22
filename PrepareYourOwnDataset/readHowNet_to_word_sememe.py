# -*- coding: utf-8 -*-
import json

sememe_inuse_Zh = json.load(open('./sememe.json'))

f=open('./HowNet_original_new.txt','r',encoding='utf-8')
endFlag = [':','}','\"']
word_sememes = {}
sememe_list = []

for line in f:
    if line[:3] == 'W_E':
        word = line[:-1].split('=')[1]
        if word not in word_sememes:
            word_sememes[word] = []
    if line[:3] == 'DEF':   
        DEF = line
        begin = False
        sememes = []
        sememe = ''
        for x in DEF:
            if (x in endFlag) and begin:
                begin = False
                sememes.append(sememe)
                sememe = ''
                continue
            if begin:
                sememe += x
            if x=='{': #x=='|':
                begin = True
        if sememes not in word_sememes[word]:
            word_sememes[word].append(sememes)
            sememe_list += sememes
    else:
        continue

f.close()

f=open('./word_sememe.txt','w',encoding='utf-8')

for w in word_sememes:
    w_lower = w.lower()
    f.write(w_lower+'||') # Caution: must use lower!!!
    count = len(word_sememes[w])
    tmp = []
    for i in range(count):
        tmp += word_sememes[w][i]
    #tmp = list(set(tmp))
    tmp = set(tmp)
    for i in tmp:
        if '|' in i:
            if i.split('|')[1] in sememe_inuse_Zh:
                f.write(i.split('|')[0]+'|')
    f.write('\n')
f.close()
