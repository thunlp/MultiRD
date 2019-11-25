# MultiRD
Code and data of the AAAI-20 paper "Multi-channel Reverse Dictionary Model"

A reverse dictionary takes the description of a target word as input and outputs the target word together with other words that match the description. 

## Requirements
* Python 3.x
* Pytorch 1.x
* Other requirements: numpy, tqdm, nltk, gensim, thulac

## Quick Start
Download the code and data from [Google Drive](https://drive.google.com/drive/folders/1jeyPE8iGdGUSVJe_6Smr_NzoWfR52f4g?usp=sharing). The code on Google drive is the same as those here.
Unzip the data.zip (under English and Chinese paths respectively), and all files under EnglishReverseDictionary and ChineseReverseDictionary paths should be prepared as follows:
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
   |- proc_allFeatures.py
   |- get_wordnet_lexname.py
   |- get_wordnet_500sample.py
   |- process_googleVec_checkAllData.py
   |- readHowNet_to_word_sememe.py
   |- wordnik_get_defi.py
   |- check_root_affix.py
```

### Training English model
Execute this command under code path：<br>
```bash
python main.py -b [batch_size] -e [epoch_num] -g [gpu_num] -sd [random_seed] -f [freq_mor] -m [rsl, r, s, l, b] -v
```
In `-m [rsl, r, s, l, b]`, `-m r` indicates the use of Morpheme information include roots and affixes. You can filter morphemes by `-f`, usually 15~35. `-m s` means to use Sememe predictor. `-m l` means to use lexnames, which is Word Category information (include Lexical name and POS information). `-m b` means not using any other information, just the basic BiLSTM model. `-m rsl` means to use all information which is our Multi-channel model. <br>
It is about 2.5G RAM when you set `-b 256` in Multi-channel model (rsl mode). <br>
`-e` usually 10 to 20. <br>
`-g` means which GPU card to use. <br>
`-v` means showing progess bar. <br>


After training, you can get two new files, `xxx_label_list.json` and `xxx_pred_list.json`. xxx means the mode you set in `-m`, e.g. the `-m rsl` setting indicates that the file will be `rsl_label_list.json`. <br>

#### Evaluation and results
Execute this command under code path：<br>
```bash
python evaluate_result.py -m [mode]
```
Here, `mode` is the same as above.<br>

Then you'll get `median rank, accuracy@1/10/100, rank variance` results on 3 testsets include **seen**, **unseen**, **description**. <br>

Model|**Seen** Definition|**Unseen** Definition|**Description**
---|:---:|:---:|:---:
OneLook|0 .66/.94/.95 200| - - - |5.5 **.33**/.54/.76 332
BOW| 172 .03/.16/.43 414 |248 .03/.13/.39 424 |22 .13/.41/.69 308
RNN| 134 .03/.16/.44 375 |171 .03/.15/.42 404 |17 .14/.40/.73 274
RDWECI| 121 .06/.20/.44 420 |170 .05/.19/.43 420 |16 .14/.41/.74 306
SuperSense| 378 .03/.15/.36 462 |465 .02/.11/.31 454 |115 .03/.15/.47 396
MS-LSTM |**0 .92/.98/.99 65**| 276 .03/.14/.37 426 |1000 .01/.04/.18 404
BiLSTM |25 .18/.39/.63 363| 101 .07/.24/49 401| 5 .25/.60/.83 214
BiLSTM+Mor| 24 .19/.41/.63 345 |80 .08/.26/.52 399 |4 .26/.62/.85 **198**
BiLSTM+Cat |19 .19/.42/.68 309 |68 .08/.28/.54 362 |4 .30/.62/.85 206
BiLSTM+Sem |19 .19/.43/.66 349| 80 .08/.26/.53 393| 4 .30/.64/.87 218
Multi-channel| 16 .20/.44/.71 310| **54 .09/.29/.58 358** |**2** .32/**.64/.88** 203

Evaluating model performance with prior knowledge. <br>
```bash
python analyse_result.py
python result_analysis_En_1200.py -m [mode]
```

Prior Knowlege|**Seen** Definition|**Unseen** Definition|**Description**
---|:---:|:---:|:---:
None |16 .20/.44/.71 310| 54 .09/.29/.58 358 |2.5 .32/.64/.88 203
POS Tag| 13 .21/.45/.72 290 |45 .10/.31/.60 348 |3 .35/.65/.91 174
Initial Letter| 1 .39/.73/.90 270 |4 .26/.63/.85 348| 0 .62/.90/.97 160
Word Length |1 .40/.71/.90 269 |6 .25/.56/.84 346 |0 .55/.85/.95 163

<br>
<br>

### Training Chinese model
Execute this command under code path：<br>
```bash
python main.py -b [batch_size] -e [epoch_num] -g [gpu_num] -sd [random_seed] -u/s -m [CPsc, C, P, s, c, b] -v
```
Different from EnglishRD training command, we use `-u` or `-s` to represent **Unseen** or **Seen** test mode. In fact, there is no need to use the test mode of Seen Definition. <br>
In `-m [CPsc, C, P, s, c, b]`, `-m C` means to use Cilin information. We use 4 Word-Classes in Cilin. `-m P` means to use POS predictor. `-m s` means to use Sememe predictor. `-m c` indicates the use of Morpheme predictor. Morphemes are characters in Chinese. `-m b` means not using any other information, just the basic BiLSTM model. `-m CPsc` means to use all information which is our Multi-channel model. <br>
It is about 7.6G RAM when you set `-b 128` in `CPsc` mode. <br>
`-e` usually 10 to 20. <br>
`-g` means which GPU card to use. <br>
`-v` means showing progess bar. <br>

#### Evaluation and results

```bash
python evaluate_result.py -m [mode]
```
Here, the `mode` is the prefix of `xxx_label_list.json`. <br>
Then you'll get `median rank, accuracy@1/10/100, rank variance` results on 4 testsets include **seen**, **unseen**, **description** and **Question**. <br>

Model |**Seen** Definition |**Useen** Definition| **Description**| **Question**
---|:---:|:---:|:---:|:---:
BOW| 59 .08/.28/.56 403| 65 .08/.28/.53 411 |40 .07/.30/.60 357 |42 .10/.28/.63 362
RNN |69 .05/.23/.55 379 |103 .05/.21/.49 405 |79 .04/.26/.53 361 |56 .07/.27/.60 346
RDWECI| 56 .09/.31/.56 423| 83 .08/.28/.52 436 |32 .09/.32/.59 376| 45 .12/.32/.61 384
BiLSTM |4 .28/.58/.78 302 |14 .15/.45/.71 343 |13 .14/.44/.78 233 |4 .30/.61/.82 243
BiLSTM+POS |4 .28/.58/.78 309| 14 .16/.45/.71 346| 13 .14/.44/.79 255| 5 .25/.59/.79 271
BiLSTM+Mor |1 .43/.73/.87 260 |11 .19/.47/.73 332 |8 .22/.52/.83 251 |1 .42/.73/.86 227
BiLSTM+Cat |4 .29/.58/.78 319 |16 .14/.43/.70 356 |13 .16/.45/.77 289 |3 .33/.62/.82 246
BiLSTM+Sem |4 .29/.60/.80 298 |14 .16/.45/.72 340 |12 .15/.45/.75 244 |4 .34/.61/.83 231
Multi-channel| 1 .49/.78/.90 220 |10 .18/.49/.76 310| 5 .24/.56/.82 260| 0 .50/.73/.90 223

Evaluating model performance with prior knowledge. <br>
```bash
python result_analysis_Ch.py -m [mode]
```

Prior Knowledge |**Seen** Definition |**Useen** Definition| **Description**| **Question**
---|:---:|:---:|:---:|:---:
None |1 .49/.78/.90 220 |10 .18/.49/.76 310 |5 .24/.56/.82 260 |0 .50/.73/.90 223
POS Tag| 1 .50/.79/.90 222 |9 .18/.51/.77 307 |4 .24/.61/.85 252 |0 .50/.74/.90 223
Initial Char| 0 .74/.89/.92 220| 0 .55/.82/.86 304| 0 .61/.88/.93 239| 0 .84/.95/.95 213
Word Length |0 .54/.82/.91 217 |6 .23/.57/.81 297 |3 .32/.68/88 242 |0 .62/.85/.94 212

## Prepare your own data
Todo...
### Data formats
Todo...
### Data process
Todo...

## Cite
```
@inproceedings{
   waiting...
}
```
