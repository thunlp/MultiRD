from wordnik import *
import time
import re

apiUrl = 'http://api.wordnik.com/v4'
apiKey = '94c3e2ed22d082fd501070c13d70f57434cdad8a57eaa5475'
client = swagger.ApiClient(apiKey, apiUrl)
wordApi = WordApi.WordApi(client)

words = {'garage', 'office', 'potato', 'facts', 'road', 'wing', 'flea', 'tallest', 'tickle', 'island', 'realize', 'clothing', 'star', 'blanket', 'allow', 'safer', 'washed', 'grow', 'toast', 'ideas', 'write', 'contain'}
f = open('./add_to_definitions.txt', 'w')
word_def = {}
i = 0
for wd in words:
    i += 1
    print(i)
    time.sleep(20) # if too little --> urllib.error.HTTPError: HTTP Error 429: Too Many Requests
    definitions = wordApi.getDefinitions(wd, sourceDictionaries='all') 
    count = 0
    for defi in definitions:
        if wd not in word_def:
            word_def[wd] = []
        if defi.text != None: # sometimes None
            tmp = re.sub('[^a-zA-Z ]', '', (defi.text).lower())
            word_def[wd].append(tmp)
            f.write(wd+' '+tmp+'\n')
            count += 1
            if count>5: # no need too much
                break
f.close()
