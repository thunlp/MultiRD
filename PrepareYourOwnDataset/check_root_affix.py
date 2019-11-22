import operator

root_affix_file = './word_root_affix.txt'
lines = open(root_affix_file, encoding='utf-8').readlines()
all_root_affix = {}
for line in lines:
    for r_a in line.strip().split():
        if r_a not in all_root_affix:
            all_root_affix[r_a] = 0
        all_root_affix[r_a] += 1
print('all_root_affix: ', len(all_root_affix)) # 27883

all_root_affix_sorted = sorted(all_root_affix.items(), key=operator.itemgetter(1),reverse=True) # 1~2031
freq_count = [0]*(31+1)
f = open('root_affix_freq.txt', 'w')
for r_a, freq in all_root_affix_sorted:
    f.write(r_a+' '+str(freq)+'\n')
    if freq>30:
        freq_count[31] += 1
    else:
        freq_count[freq] += 1
f.close()


f = open('root_affix_freq_statistics.csv', 'w')
for i in range(len(freq_count)):
    f.write(str(i)+';'+str(freq_count[i])+'\n')
f.close()