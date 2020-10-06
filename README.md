# MultiRD
Code and data of the AAAI-20 paper "**Multi-channel Reverse Dictionary Model**" [[pdf](https://arxiv.org/pdf/1912.08441.pdf)]

## Requirements
* Python 3.x
* Pytorch 1.x
* Other requirements: numpy, tqdm, nltk, gensim, thulac

## Quick Start
Download the code and data from [Google Drive](https://drive.google.com/drive/folders/1jeyPE8iGdGUSVJe_6Smr_NzoWfR52f4g?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ec29131d38fd4ca2a6ca/), where the code is the same as that here.

Unzip the data.zip (under English and Chinese paths respectively), and all files under `EnglishReverseDictionary` and `ChineseReverseDictionary` should be prepared as follows:

```
ReverseDictionary
|- EnglishReverseDictionary
|  |- data
|  |  |- data_train.json
|  |  |- data_dev.json
|  |  |- data_test_500_rand1_seen.json
|  |  |- data_test_500_rand1_unseen.json
|  |  |- data_defi_c.json           [definitions of the target words in 200 descriptions]
|  |  |- data_desc_c.json           [testset of 200 descriptions]
|  |  |- vec_inuse.json             [Only embeddings used in this model are included.]
|  |  |- lexname_all.txt
|  |  |- root_affix_freq.txt
|  |  |- sememes_all.txt
|  |  |- target_words.txt
|  |- code
|     |- main.py
|     |- model.py
|     |- data.py
|     |- evaluate.py
|     |- evaluate_result.py
|     |- analyse_result.py
|     |- result_analysis_En_1200.py
|- ChineseReverseDictionary
|  |- data
|  |  |- Cx.json                    [x=1,2,3,4]
|  |  |- description_sense.json     [train & dev dataset]
|  |  |- description_idio_locu.json [testset of Question]
|  |  |- description_byHand.json    [testset of description]
|  |  |- hownet.json
|  |  |- sememe.json
|  |  |- word_cilinClass.json
|  |  |- word_index.json
|  |  |- word_vector.npy            [Only embeddings used in this model are included.]
|  |- code
|     |- main.py
|     |- model.py
|     |- data.py
|     |- evaluate.py
|     |- evaluate_result.py
|- PrepareYourOwnDataset
   |- <See below.>
```

### Train English Model
Execute this command under code path：
```bash
python main.py -b [batch_size] -e [epoch_num] -g [gpu_num] -sd [random_seed] -f [freq_mor] -m [rsl, r, s, l, b] -v
```
In `-m [rsl, r, s, l, b]`, 

- `-m r` indicates the use of Morpheme information including roots and affixes. You can filter morphemes by `-f`, usually 15~35;
-  `-m s` means using the Sememe predictor;
-  `-m l` means using WordNet lexnames, which is word category information (include Lexical name and POS tag information);
-  `-m b` means not using any other information, just the basic BiLSTM model;
-  `-m rsl` means to use all information which is our Multi-channel model;

`-e` is usually set to 10~20;

`-g` indicates which GPU to use;

`-v` means showing progess bar.


After training, you will get two new files, `xxx_label_list.json` and `xxx_pred_list.json`. "xxx" indicates the mode you set in `-m`, e.g., the `-m rsl` setting indicates that the file will be `rsl_label_list.json`. 

#### Evaluation
Execute this command under code path:
```bash
python evaluate_result.py -m [mode]
```
Here, `mode` is the same as above.

Then you'll get `median rank`,  ` accuracy@1/10/100` and  `rank variance` results on 3 test sets including **seen**, **unseen** and **description**. 



You can evaluate model performance with prior knowledge:

```bash
python analyse_result.py
python result_analysis_En_1200.py -m [mode]
```

### Train Chinese Model
Execute this command under code path：
```bash
python main.py -b [batch_size] -e [epoch_num] -g [gpu_num] -sd [random_seed] -u/-s -m [CPsc, C, P, s, c, b] -v
```
Different from English model training, we use `-u` or `-s` to represent **Unseen** or **Seen** test mode. In fact, there is no need to use the test mode on the Seen Definition test set. 
In `-m [CPsc, C, P, s, c, b]`

-  `-m C` means using Cilin word category information and we use 4 word classes in Cilin;
-  `-m P` means using POS predictor;
-  `-m s` means using Sememe predictor;
-  `-m c` indicates the use of Morpheme predictor where morphemes are Chinese characters;
-  `-m b` means not using any other information, just the basic BiLSTM model;
-  `-m CPsc` means to use all information as our Multi-channel model.

`-e` , `-g` and `-v` are the same as those in English model training. 

#### Evaluation

```bash
python evaluate_result.py -m [mode]
```
Here, the `mode` is the prefix of `xxx_label_list.json`. 
Then you'll get `median rank`,  ` accuracy@1/10/100` and  `rank variance` results on 4 test sets including **seen**, **unseen**, **Description** and **Question**. 



You can evaluate model performance with prior knowledge:
```bash
python result_analysis_Ch.py -m [mode]
```

## Prepare Your Own Data

Here is some code for reference. The data format is shown below, and you can build your own data set.
```
ReverseDictionary
|- EnglishReverseDictionary
|- ChineseReverseDictionary
|- PrepareYourOwnDataset
   |- proc_allFeatures.py
   |- get_wordnet_lexname.py
   |- get_wordnet_500sample.py
   |- process_googleVec_checkAllData.py
   |- readHowNet_to_word_sememe.py
   |- wordnik_get_defi.py
   |- check_root_affix.py
```
### Data Formats
It is json format in data_xxx.json files.
```
{
     "word": "fatalism",
     "lexnames": [
         "noun.cognition"
     ],
     "root_affix": [
         "fatal",
         "ism"
     ],
     "sememes": [
         "knowledge",
         "believe",
         "experience",
         "Fate"
     ],
     "definitions": "the doctrine that all events are predetermined by fate and are therefore unalterable"
}
```
Word embeddings are in `vec_inuse.json` which contains all target words and words in definitions. Only used words are included. The format is `{word: [vector]}`, ....
`lexname_all.txt` contains all 45 lexnames from WordNet.
`sememes_all.txt` contains 1400 sememes from HowNet.
Morphemes (root and affix) are in `root_affix_freq.txt`, which contains morphemes and their numbers, separated by spaces.

### Download and Process Data
In English experiments, we use the Description dataset from [(Hill et al. 2016)](https://arxiv.org/pdf/1504.00548.pdf). 

Word embeddings are from [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). 

Sememes can be obtained using [OpenHowNet](https://github.com/thunlp/OpenHowNet). 

Lexnames are from WordNet which you can get them easily by NLTK.

We get morphemes by [Morfessor tool](https://morfessor.readthedocs.io/en/latest/). The used dataset is from [morpho.aalto.fi](http://morpho.aalto.fi/events/morphochallenge2010/datasets.shtml). You should train mofessor model first, and then use it to process the target words to get the corresponding roots and affixes.

```bash
morfessor-train --encoding=ISO_8859-15 --traindata-list --logfile=log.log -s model.bin -d ones wordlist-2010.eng
morfessor-segment -l ../morfessor_data/model.bin target_words.txt -o word_root_affix.txt
```
Unfortunately, the morphemes obtained by this method are not accurate. It is recommended that you use the standard root-affix dictionary.



## Cite
If you use any code or data, please cite this paper

```
@article{zhang2019multi
    title={Multi-channel Reverse Dictionary Model},
    author={Zhang, Lei and Qi, Fanchao and Liu, Zhiyuan and Wang, Yasheng and Liu, Qun and Sun, Maosong},
    journal={arXiv preprint arXiv:1912.08441},
  	year={2019}
}
```

## Contact
You can visit our [online reverse dictionary website](https://wantwords.thunlp.org/), where we have optimized our methods and datasets. Github [WantWords](https://github.com/thunlp/WantWords). You can post issues if you have any questions.
