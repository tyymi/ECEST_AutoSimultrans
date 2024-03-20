# AutoSimultrans 

##  Background：

Combining artificial intelligence technologies such as machine translation (MT), automatic speech recognition (ASR) and text-to-speech synthesis (TTS), simultaneous interpretation has developed into a cutting-edge research field with a wide range of applications in numerous scenarios, including international conferences, business negotiations, news presentations, legal proceedings and medical communication. As an emerging interdisciplinary field, simultaneous interpreting will face more challenges in the future.

In order to promote the development of machine simultaneous interpretation technology, Baidu successfully bid for the 3rd Symposium on Simultaneous Interpretation at the NAACL Top Meeting, which brought together many researchers and practitioners in the fields of machine translation, speech processing and human interpretation to discuss the latest advances in simultaneous interpretation and the outstanding challenges faced today.

## About this repo

This repo is based on PaddlePaddle framework, using bpe-based Transformer as the translation model and waitk strategy for simultaneous translation, and achieved the second place in [AutoSimulTrans22](https://aistudio.baidu.com/aistudio/competition/detail/148) in the English version of the track.

## Dataset


| Dataset | Language-pair | Size |
|-------|-------|-------|
| [CWMT21](http://mteval.cipsc.org.cn:81/agreement/AutoSimTrans) | Zh-En  | 9.1m |
| CWMT21 | Zh mono  | 1m |
| BSTC | Zh-En | 3.7w |
| 	[UN Parallel Corpus](https://conferences.unite.un.org/UNCORPUS/en/DownloadOverview#download) |  En-Es | 21m |

## Model

The top of Figure 3 is the classic seq2seq architecture, where the entire source is input to generate the target; while the bottom is the prefix2prefix architecture, where only part of the source sentence prefix is used to generate the target word:

![seq2seq/prefix2prefix](./images/compare.png)

The architectureof the translation model for this project is shown below, with only the number of encoders in the transformer base changed to 12.

| Configuration      | Value |
| ------------------ | ----- |
| Encoder depth      | 12    |
| Decoder depth      | 6     |
| Attention heads    | 8     |
| Embedding dim      | 512   |
| FFN size           | 2048  |
| Chinese vocab size | 45942 |
| English vocab size | 32151 |
| dropout            | 0.1   |

Besides, we use methods in [DeepNet](https://arxiv.org/abs/2203.00555) that modifies the residual connection and a theoretically derived initialization.

![](https://ai-studio-static-online.cdn.bcebos.com/a39f5757965d4016884540d1f7ed79ea8cdf96ed5cf64dd4becbc20e72444971)

​		In order to evaluate the degree of implementation of the model, training, evaluation, etc. in this project, a comparison was made with fairseq, training base and the deep encoder used in this repo on a 2m ccmt, with the following parameters and results.

| lr   | warmup | optimizer      | schedule     | update-freq | dropout |
| ---- | ------ | -------------- | ------------ | ----------- | ------- |
| 5e-4 | 4000   | adam(0.9,0.98) | inverse_sqrt | 4           | 0.1     |

| Frame   | Arch           | Epoch | Bleu        | Speed（steps/s） |
| ------- | -------------- | ----- | ----------- | ---------------- |
| fairseq | base           | 16    | 23.08       | 10.5(3090)       |
| fairseq | big            | -     | -           | -                |
| paddle  | base           | 7     | **23.1846** | 3.4 （V100）     |
| paddle  | 12+6+deepnorm√ | 17    | 23.1153     | 2.8 （V100）     |
| paddle  | big            | -     | -           | 1.4 （V100）     |

​	We observed that the Paddle version of the Transformer performed slightly better the fairseq version. Aside from that, the Transformer base seems to outperform our implementation of deepnorm, probably due to the size of the dataset. 

​	

## Quick start

### 1.Preparation

```shell
git clone https://github.com/MiuGod0126/STACL_Paddle.git
cd STACL_Paddle
pip install -r requirements
```

### 2.Code Structure

```
├── ckpt 
├── configs
├── dataset
│   ├── ccmt21
│   ├── bstc
│   ├── enes21
├── decode
├── examples 
├── models
├── reader
├── paddleseq_cli 
│   ├── preprocess.py
│   ├── train.py 
│   ├── valid.py 
│   ├── generate.py 
│   ├── config.py 
├── scripts 
├── tools
├── output 
├── requirements.txt 
├── README.md
```

### 3.Data Processing

#### 3.1 Preprocessing

- Word segmentation: for Chinese first use jieba splitting; then use moses normalize-punctuation and tokenizer for Chinese and English (Western) respectively.(In fact Chinese does not need to use moses, while moses needs de-tokenizing after decoding).
- Length filtering: for Chinese-English, filter out parallel corpus with length 1-250 and length ratio more than 1:2.5 or 2.5:1; for English-Spanish, filter out parallel corpus with length 1-250 and length ratio more than 1:1.5 or 1.5:1.
- Language id filtering (lang id): use fasttext to filter out parallel text that does not match the language id on either side of the source or target.
- Deduplication: For the Chinese monolingual, de-duplication was performed and reduced by 3m.
- Truecase： We use truecase for both English and Spanish languages to automatically determine when names, places, etc. in a sentence are selected in case form rather than directly using lowercase, and de-truecaseing is required after decoding.(Chinese is not used, and this step requires training the model and is very time-consuming to process).
- BPE(byte-pair-encoding): For Chinese-English, each uses 32K operations; for English-Western, a shared 32K subword word list; where the Chinese->English word list contains the training set of ccmt, bstc, and the monolingual Chinese corpus of ccmt.

#### 3.2 Binarize

The repo supports two formats of data input, a text pair, and fairseq binary data (which can be compressed by half). To generate bin data for bstc, for example, the command is as follows (see [this](#bin_load) for the use of bin data).	

```shell
workers=1
TEXT=dataset/bstc
python paddleseq_cli/preprocess.py \
        --source-lang zh --target-lang en \
        --srcdict $TEXT/vocab.zh --tgtdict  $TEXT/vocab.en \
        --trainpref $TEXT/asr.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/dev.bpe  \
        --destdir data_bin/bstc_bin --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers
#⭐or
bash scripts/preprocess.sh
```

The results are shown below:

```
data_bin/bstc_bin/
    preprocess.log
    test.zh-en.en.idx
    test.zh-en.en.bin
    test.zh-en.zh.bin
    train.zh-en.en.bin
    test.zh-en.zh.idx
    train.zh-en.zh.bin
    train.zh-en.en.idx
    train.zh-en.zh.idx
    valid.zh-en.en.bin
    valid.zh-en.en.idx
    valid.zh-en.zh.bin
    valid.zh-en.zh.idx
```

**Note: workers>1 is supported on windows, while workers=1 is currently only available on aistudio**

### 4.full-sentence training

Taking the provided Chinese-English ccmt translation data as an example, the following commands can be executed to train the model.

```shell
# Single or multi-card training (set ngpus)
ngpus=4
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml \
                         --amp \
                         --ngpus $ngpus  \
                         --update-freq 4 \
                         --max-epoch 10 \
                         --save-epoch 1 \
                         --save-dir /root/paddlejob/workspace/output \
                         --log-steps 100 \
                         --max-tokens 4096 \
#⭐ or
bash scripts/train_full.sh
# Model validation
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml  --pretrained ckpt/model_best_zhen --eval
```

For Chinese-English after training on ccmt, fine-tuning with zhen_bstc.yaml is also needed:

```
├── configs
│   ├── enes_un.yaml 
│   ├── enes_waitk.yaml 
│   ├── zhen_ccmt.yaml 
│   ├── zhen_bstc.yaml 
│   ├── zhen_bstc_bin.yaml 
│   ├── zhen_waitk.yaml  

```

In addition to this, there are two approaches when the amount of data is too large.

<a id="bin_load"></a>

1. Partial training: modify **train.train_data_size** in the configuration file, default -1 means load all. Suitable for those who need fast loading debugging, or fine-tuning the model with a small amount of corpus.
2. ⭐Partial load : Use iterator to get a pool size data first, then use MapDataset to load dynamic group batch in full, which greatly improves data loading speed and prevents memory explosion. To use this feature, first use the command in data preparation to generate binary data, and then modify the configuration file **data.use_binary** and **data.lazy_load** to True (don't forget to modify the data prefix), see **zhen_bstc_bin.yaml** for details, the training command remains unchanged.



### 5.Prediction

Taking ccmt21 as an example, the following commands can be executed to translate the text in the specified file after the model is trained, and the results are output to output/generate.txt by default.

```shell
python  paddleseq_cli/generate.py --cfg configs/zhen_ccmt.yaml \
				   --pretrained ckpt/model_best_zhen \
				   --beam-size 5 \
				   --generate-path generate.txt \
				   --sorted-path result.txt
				   # --only-src # 

#⭐ or
bash scripts/generate_full.sh
```

Training, validation curves were generated using visualdl:

```shell
visualdl --logdir output/vislogs/zhen --port 8080
# 打开链接：localhost:8080
```

### 6.waitk training

After training on the dataset of ccmt 9m, then fine-tuning on bstc (zhen_bstc.yaml), and finally fine-tuning the homography model on bstc with waitk, the command is as follows:

```shell
k=5 #full sentence k=-1
python paddleseq_cli/train.py --cfg configs/zhen_waitk.yaml \
            --waitk $k --pretrained ckpt/model_best_zhen

# ⭐or
bash scripts/train_wk.sh
```

### 7.stream prediction

```shell
k=5
stream_prefix=dataset/stream_zh/dev/3
ckpt_dir=model_best_zhen 
python paddleseq_cli/generate.py --cfg configs/zhen_waitk.yaml \
            --test-pref $stream_prefix --only-src \
            --pretrained  $ckpt_dir \
            --waitk $k --stream \
            --infer-bsz 1 --beam-size 5
# ⭐ or
bash scripts/generate_wk.sh
```

```
############## input ############## 
额 ，
额 ， 非
额 ， 非常
额 ， 非常 非
额 ， 非常 非常
额 ， 非常 非常 荣
额 ， 非常 非常 荣幸
额 ， 非常 非常 荣幸 能
额 ， 非常 非常 荣幸 能@@ 今
额 ， 非常 非常 荣幸 能 今天
############## output ############## 
Well
,

 it
 is
 a

 great
 honor


 to
 be

 here
```

### 8.waitk evaluation

dataset/Zh-En/dev contains multiple files, use scripts to predict and evaluate bleu and al on dev files in one click, Chinese and English generally takes more than 10 minutes.

```shell
# chinest-to-english
k=5
ckpt_dir=<ckpt_dir>
beam_size=1
bash scripts/gen_eval_zhen.sh dev $k $ckpt_dir $beam_size
# english-to-spanish
bash scripts/gen_eval_enes.sh dev $k $ckpt_dir $beam_size
```

**Note: Due to the Chinese and English prediction of dozens of files, and memory and arithmetic power to run is not enough, I try shell multi-process run waitk prediction, the speed can be reduced to about 5min, but the accuracy is significantly reduced, do not know the reason for the time being, the command is as follows (gen_eval_zhen_paral.sh):**

```shell
# chinest-to-english
k=5
ckpt_dir=<ckpt_dir>
beam_size=1
workers=2
bash scripts/gen_eval_zhen_paral.sh dev $k $ckpt_dir $beam_size $workers
```

###  9.waitk predict

dataset/Zh-En/dev contains multiple files, use scripts to predict and evaluate bleu and al on dev files with one click.

```shell
# chinest-to-english
k=5
ckpt_dir=<ckpt_dir>
beam_size=1
bash scripts/gen_eval_zhen.sh test $k $ckpt_dir $beam_size
# english-to-spanish
bash scripts/gen_eval_enes.sh test $k $ckpt_dir $beam_size
```

###  10.back translation

1. (X,Y)  train forward model F

   ```shell
   python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml --src-lang zh --tgt-lang en 
   ```

2. (Y,X) train backward model B

   ```shell
   python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml --src-lang en --tgt-lang zh
   ```

3. average checkpoints

   ```shell
   k=3
   python scripts/average_checkpoints.py \
   	--inputs output/ckpt  \
   	--output output/ckpt/avg${k} --num-ckpts $k
   ```
   
4. 单语Y1分片（当数据太大时，分不同机器预测）

   ```shell
   workers=2
   infile= dataset/mono.en
   bash examples/backtrans/shard.sh $workers $infile
   ```

5. Model B predicts X1

   ```shell
   ckpt_dir=model_best_enzh
   mono_file=dataset/mono.en
   python paddleseq_cli/generate.py --cfg configs/zhen_ccmt.yaml \
   			--src-lang en --tgt-lang zh \
               --test-pref $mono_file --only-src \
               --pretrained  $ckpt_dir  --remain-bpe
   ```
   
6. See the predicted result logprob distribution:

   Influenced by the quality of the inverse model B, the generated results may be poor, as reflected by the low lprob score in generate.txt. The lprob distribution (which can be used to set the filtering threshold min-lprob at 7 extraction) can be viewed using the following command:

   ```shell
   python examples/backtrans/lprob_analysis.py output/generate.txt
   ```

   Results is:

   ```
               lprobs
   count  4606.000000
   mean     -1.060325
   std       0.256854
   min      -2.578100
   25%      -1.225675
   50%      -1.054400
   75%      -0.890825
   max      -0.209400
   ```

7. extract parallel data P' (X1,Y1)

   ```shell
   python examples/backtrans/extract_bt_data.py \
   		--minlen 1 --maxlen 250 --ratio 2.5 --min-lprob -3 \
   		--output output/ccmt_ft --srclang zh --tgtlang en  \
   		 output/bt/generate*.txt
   ```
   
8. merge data (X,Y) (X1,Y1) and continue to train F



## Reference

[1. STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://www.aclweb.org/anthology/P19-1289.pdf)

[2.SimulTransBaseline](https://aistudio.baidu.com/aistudio/projectDetail/315680/)：

[3.PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/simultaneous_translation/stacl)

[4.fairseq](https://github.com/pytorch/fairseq)

[5.ConvS2S_Paddle](https://github.com/MiuGod0126/ConvS2S_Paddle)

[6.DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

