import random
# random.randint(0, n)

def load_filterlist(path):
    with open(path) as input_file:
        filterlist = input_file.read().splitlines()
    return set(filterlist)   
target_words = load_filterlist('./target_words.txt')  

lines = open('wordnet_data_all.txt').readlines()
random.shuffle(lines)

target_words_defi_500_rand1_unseen = {}
target_words_defi_500_rand1_seen = {}
target_words_defi_500_all_seen = {}
i = 0
count = 0
for line in lines:
    [word, contents] = line.strip().split('|||')
    if word in target_words:
        i += 1
        if i>1000:
            break
        defi = []
        for con in contents.split('||'):
            defi.append(con.split('|')[2])
        if i<=500:
            target_words_defi_500_rand1_unseen[word] = defi.pop(random.randint(0, len(defi)-1)) # very nice method, i think
        else:
            target_words_defi_500_all_seen[word] = defi.copy() # must copy(), if not, this will change after defi.pop()!!!!!!!
            count += len(defi)
            target_words_defi_500_rand1_seen[word] = defi.pop(random.randint(0, len(defi)-1)) # very nice method, i think
    

with open('wordnet_defi_500_rand1_unseen.txt', 'w') as f:
    for wd in target_words_defi_500_rand1_unseen:
        f.write(wd+' ')
        f.write(target_words_defi_500_rand1_unseen[wd])
        f.write('\n')
with open('wordnet_defi_500_rand1_seen.txt', 'w') as f:
    for wd in target_words_defi_500_rand1_seen:
        f.write(wd+' ')
        f.write(target_words_defi_500_rand1_seen[wd])
        f.write('\n')
coun = 0
with open('wordnet_defi_500_all_seen.txt', 'w') as f:
    for wd in target_words_defi_500_all_seen:
        for defi in target_words_defi_500_all_seen[wd]:
            f.write(wd+' ')
            f.write(defi)
            f.write('\n')
            coun += 1
assert coun == count
print('saved target_words_defi_500_rand1 to file.')
