# Tutorials for JIIST Experiments
Journal of Intelligent Informatics and Smart Technology

- [Machine Learning approach](#Machine-Learning-approach)
  - [CRF model](#CRF-model)
  - [RDR model](#RDR-model)
  - [HMM model](#HMM-model)
 - [Neural Machine Translation approach](#Neural-Machine-Translation-approach)
   - [Seq2Seq model](#Seq2Seq-model)
   - [Transformer model](#Transformer-model)
 - [Evaluation](#Evaluation)
   - [WER Calculation](#WER-Calculation)
   - [chrF Calculation](#chrF-Calculation)
   - [BLEU Score Calculation](#BLEU-Score-Calculation)
 
 ## Model guide
 - sent-models: models trained on sentence-only train data [[sent-models](sent-models)]
 - sent+para-models: models trained on sentence+paragraph train data [[sent+para-models](sent+para-models)] 

 ## Machine Learning approach
 
 ### CRF model
Conditional Random Fields <br /> 
Step 1: Download CRF++ Tool (CRF++-0.58.tar.gz)
 
 ```
 https://drive.google.com/drive/u/0/folders/0B4y35FiV1wh7fngteFhHQUN2Y1B5eUJBNHZUemJYQV9VWlBUb3JlX0xBdWVZTWtSbVBneU0
 ```
 
Step 2: Read dependencies for installation for CRF++ in detail and Install
 
 ```
$ ./configure 
$ make
$ sudo make install
 ```
 
Step 3: Check to use 
 
 ```
$ crf_learn --help
$ crf_test --help
 ```
 
Step 4: Training and Testing CRF++ with CRF format data in "./data/" directory
 ```
$ cat template
  # Unigram
  U00:%x[-2,0]
  U01:%x[-1,0]
  U02:%x[0,0]
  U03:%x[1,0]
  U04:%x[2,0]

  # Bigram
 ```
 
 ``` 
$ crf_learn ./template train.col ./sent_level.crf-model | tee sent_crf.train.log
 ```

```
$ crf_test -m ./sent_level.crf-model test.col > ./result.col
$ head -5 result.col
  အခု		B	B
  သန့်စင်ခန်း	N	O
  ကို		N	N
  သုံး		N	N
  ပါရစေ		E	E
```
First column is text, second column is ground-truth label and third column is model prediction. Therefore cut "text" and "prediction" only.
```
$ cut -f1,3 ./result.col > ./result.col.f13
$ head -5 result.col.f13
  အခု		B
  သန့်စင်ခန်း	O
  ကို		N
  သုံး		N
  ပါရစေ		E
```
Convert column format to line (col2line.pl is in the program folder.)
```
$ perl col2line.pl result.col.f13 > result.txt
$ head -5 result.txt
  အခု/B သန့်စင်ခန်း/O ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/O မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/N မ/O ရ/O အောင်/O ကျ/N သွား/N တယ်/E
  ဂီနီ/B နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```

### RDR models

Ripple Down Rule-based POSTagger <br />
Step 1: Download RDRPOSTagger

```
$ git clone https://github.com/datquocnguyen/RDRPOSTagger
```
Step 2: Change Directory to Python port and Train with tagged data
```
$ cd pSCRDRtagger
```
Train data example ...
```
$ head -5 train.tagged 
  ဘာ/B ရယ်/O လို့/O တိတိကျကျ/O ထောက်မပြ/O နိုင်/O ပေမဲ့/O ပြဿနာ/O တစ်/O ခု/O ခု/O ရှိ/O တယ်/N နဲ့/N တူ/N တယ်/E
  လူ့/B အဖွဲ့အစည်း/O က/O ရှပ်ထွေး/O လာ/O တာ/O နဲ့/O အမျှ/O အရင်/O က/O မ/O ရှိ/O ခဲ့/O တဲ့/O လူမှုရေး/O ပြဿနာ/O တွေ/O ဖြစ်ပေါ်/N လာ/N ခဲ့/N တယ်/E
  အခု/B အလုပ်/O လုပ်/N နေ/N ပါ/N တယ်/E
  ကြည့်/B ရေစာ/O တွေ/O က/O အဲဒီ/O တစ်/O ခု/O နဲ့/N မ/N တူ/N ဘူး/E
  ဘူမိ/B ရုပ်သွင်/O ပညာ/O သည်/O ကုန်းမြေသဏ္ဌာန်/O များ/O ကို/O လေ့လာ/O သော/N ပညာရပ်/N ဖြစ်/N သည်/E
```
```
$ python2.7 RDRPOSTagger.py train train.tagged
```
Step 3: Testing <br />
Test data format example
```
$ head test.my
  အခု သန့်စင်ခန်း ကို သုံး ပါရစေ
  လူငယ် တွေ က ပုံစံတကျ ရှိ မှု ကို မ ကြိုက် ဘူး
  ဒီ တစ် ခေါက် ကိစ္စ ကြောင့် ကျွန်တော့် ရဲ့ သိက္ခာ အဖတ်ဆယ် လို့ မ ရ အောင် ကျ သွား တယ်
  ဂီနီ နိုင်ငံ သည် ကမ္ဘာ ပေါ် တွင် ဘောက် ဆိုက် တင်ပို့ မှု အများဆုံး နိုင်ငံ ဖြစ် သည်
  ဘာ လုပ် ရ မလဲ ဟင်
```
Testing ...
```
$ python2.7 RDRPOSTagger.py tag train.tagged.RDR train.tagged.DICT test.my

	=> Read a POS tagging model from train.tagged.RDR

	=> Read a lexicon from train.tagged.DICT

	=> Perform POS tagging on test.my

	Output file: test.my.TAGGED
```
RDRPOSTagger tagged data ...
```
$ head -5 test.my.TAGGED 
  အခု/B သန့်စင်ခန်း/O ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/N မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/O မ/O ရ/O အောင်/O ကျ/N သွား/N တယ်/E
  ဂီနီ/B နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```

### HMM models
3-gram Hidden Markov Models POSTagger with jita-0.3.3. <br />
Step 1: Download jita-0.3.3.

```
https://github.com/danieldk/jitar/releases
```
Step 2: Train with jita-0.3.3.
```
$ jitar-0.3.3-bin/jitar-0.3.3$ bin/jitar-train brown ./train.tagged ./sent.hmm.model | tee 3gHMM-training.log
```
Step 3: Testing ... <br/>
Test data format example
```
$ head test.my
  အခု သန့်စင်ခန်း ကို သုံး ပါရစေ
  လူငယ် တွေ က ပုံစံတကျ ရှိ မှု ကို မ ကြိုက် ဘူး
  ဒီ တစ် ခေါက် ကိစ္စ ကြောင့် ကျွန်တော့် ရဲ့ သိက္ခာ အဖတ်ဆယ် လို့ မ ရ အောင် ကျ သွား တယ်
  ဂီနီ နိုင်ငံ သည် ကမ္ဘာ ပေါ် တွင် ဘောက် ဆိုက် တင်ပို့ မှု အများဆုံး နိုင်ငံ ဖြစ် သည်
  ဘာ လုပ် ရ မလဲ ဟင်
```

```
$ cat test.my | bin/jitar-tag ./sent.hmm.model > ./test.hmm.result
```
```
$ head -5 test.hmm.result
	B N N N E
	B O O O O O O N N E
	B O O O O O O O O N N N N N N E
	O O O O O O O O O O O N N E
	B N N N E
```
Make pair format with mk-pair.pl in program folder
```
$ ./mk-pair.pl test.my test.hmm.result > test.hmm.TAGGED
```
```
$ head -5 test.hmm.TAGGED
  အခု/B သန့်စင်ခန်း/N ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/O မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/N မ/N ရ/N အောင်/N ကျ/N သွား/N တယ်/E
  ဂီနီ/O နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```

## Neural Machine Translation approach
- Experiments log details: https://github.com/ye-kyaw-thu/error-overflow/blob/master/mySent-exp4.md <br />
- For NMT experiments, we used parallel data format in the data folder. <br />
- We used **Marian** NMT tool for our NMT experiments. <br/>
- Installation Details: [[Marian Version1.10.0](https://github.com/ye-kyaw-thu/error-overflow/blob/master/marian-ver1.10.0-installation-log.md)] [[Marian Version 1.12.0](https://github.com/ye-kyaw-thu/error-overflow/blob/master/marian-1.12.0-installation-log.md)]
<br/>
- Config yml file for reference:  

```
# Marian configuration file generated at 2022-12-14 13:25:39 +0000 with version v1.11.0 f00d0621 2022-02-08 08:39:24 -0800
# General options
authors: false
cite: false
build-info: ""
workspace: 4500
log: model.seq2seq.sent1/train.log
log-level: info
log-time-zone: ""
quiet: false
quiet-translation: false
seed: 1111
check-nan: false
interpolate-env-vars: false
relative-paths: false
sigterm: save-and-exit
# Model options
model: model.seq2seq.sent1/model.npz
pretrained-model: ""
ignore-model-config: false
type: s2s
dim-vocabs:
  - 0
  - 0
dim-emb: 512
factors-dim-emb: 0
factors-combine: sum
lemma-dependency: ""
lemma-dim-emb: 0
dim-rnn: 1024
enc-type: alternating
enc-cell: lstm
enc-cell-depth: 4
enc-depth: 3
dec-cell: lstm
dec-cell-base-depth: 4
dec-cell-high-depth: 2
dec-depth: 3
skip: true
layer-normalization: true
right-left: false
input-types:
  []
best-deep: false
tied-embeddings: true
tied-embeddings-src: false
tied-embeddings-all: false
output-omit-bias: false
transformer-heads: 8
transformer-no-projection: false
transformer-pool: false
transformer-dim-ffn: 2048
transformer-decoder-dim-ffn: 0
transformer-ffn-depth: 2
transformer-decoder-ffn-depth: 0
transformer-ffn-activation: swish
transformer-dim-aan: 2048
transformer-aan-depth: 2
transformer-aan-activation: swish
transformer-aan-nogate: false
transformer-decoder-autoreg: self-attention
transformer-tied-layers:
  []
transformer-guided-alignment-layer: last
transformer-preprocess: ""
transformer-postprocess-emb: d
transformer-postprocess: dan
transformer-postprocess-top: ""
transformer-train-position-embeddings: false
transformer-depth-scaling: false
bert-mask-symbol: "[MASK]"
bert-sep-symbol: "[SEP]"
bert-class-symbol: "[CLS]"
bert-masking-fraction: 0.15
bert-train-type-embeddings: true
bert-type-vocab-size: 2
dropout-rnn: 0.3
dropout-src: 0.3
dropout-trg: 0
transformer-dropout: 0
transformer-dropout-attention: 0
transformer-dropout-ffn: 0
# Training options
cost-type: ce-sum
multi-loss-type: sum
unlikelihood-loss: false
overwrite: false
no-reload: false
train-sets:
  - /home/ye/exp/mysent/data-sent/train.my
  - /home/ye/exp/mysent/data-sent/train.tg
vocabs:
  - /home/ye/exp/mysent/data-sent/vocab/vocab.my.yml
  - /home/ye/exp/mysent/data-sent/vocab/vocab.tg.yml
sentencepiece-alphas:
  []
sentencepiece-options: ""
sentencepiece-max-lines: 2000000
after-epochs: 0
after-batches: 0
after: 0e
disp-freq: 500
disp-first: 0
disp-label-counts: true
save-freq: 5000
logical-epoch:
  - 1e
  - 0
max-length: 200
max-length-crop: false
tsv: false
tsv-fields: 0
shuffle: data
no-shuffle: false
no-restore-corpus: false
tempdir: /tmp
sqlite: ""
sqlite-drop: false
devices:
  - 0
num-devices: 0
no-nccl: false
sharding: global
sync-freq: 200u
cpu-threads: 0
mini-batch: 64
mini-batch-words: 0
mini-batch-fit: true
mini-batch-fit-step: 10
gradient-checkpointing: false
maxi-batch: 100
maxi-batch-sort: trg
shuffle-in-ram: false
data-threads: 8
all-caps-every: 0
english-title-case-every: 0
mini-batch-words-ref: 0
mini-batch-warmup: 0
mini-batch-track-lr: false
mini-batch-round-up: true
optimizer: adam
optimizer-params:
  []
optimizer-delay: 1
sync-sgd: true
learn-rate: 0.0001
lr-report: false
lr-decay: 0
lr-decay-strategy: epoch+stalled
lr-decay-start:
  - 10
  - 1
lr-decay-freq: 50000
lr-decay-reset-optimizer: false
lr-decay-repeat-warmup: false
lr-decay-inv-sqrt:
  - 0
lr-warmup: 0
lr-warmup-start-rate: 0
lr-warmup-cycle: false
lr-warmup-at-reload: false
label-smoothing: 0
factor-weight: 1
clip-norm: 1
exponential-smoothing: 0.0001
guided-alignment: none
guided-alignment-cost: mse
guided-alignment-weight: 0.1
data-weighting: ""
data-weighting-type: sentence
embedding-vectors:
  []
embedding-normalization: false
embedding-fix-src: false
embedding-fix-trg: false
fp16: false
precision:
  - float32
  - float32
cost-scaling:
  []
gradient-norm-average-window: 100
dynamic-gradient-scaling:
  []
check-gradient-nan: false
normalize-gradient: false
train-embedder-rank:
  []
quantize-bits: 0
quantize-optimization-steps: 0
quantize-log-based: false
quantize-biases: false
ulr: false
ulr-query-vectors: ""
ulr-keys-vectors: ""
ulr-trainable-transformation: false
ulr-dim-emb: 0
ulr-dropout: 0
ulr-softmax-temperature: 1
task:
  []
# Validation set options
valid-sets:
  - /home/ye/exp/mysent/data-sent/valid.my
  - /home/ye/exp/mysent/data-sent/valid.tg
valid-freq: 5000
valid-metrics:
  - cross-entropy
  - perplexity
  - bleu
valid-reset-stalled: false
early-stopping: 10
early-stopping-on: first
beam-size: 12
normalize: 0
max-length-factor: 3
word-penalty: 0
allow-unk: false
n-best: false
word-scores: false
valid-mini-batch: 32
valid-max-length: 1000
valid-script-path: ""
valid-script-args:
  []
valid-translation-output: ""
keep-best: false
valid-log: model.seq2seq.sent1/valid.log
```

### Seq2Seq model

```bash
root@57452252667f:/home/ye/exp/mysent# head -n 30 ./seq2seq.sent1.sh
#!/bin/bash

## Written by Ye Kyaw Thu, Affiliated Professor, CADT, Cambodia
## for NMT Experiments between Burmese dialects
## used Marian NMT Framework for seq2seq training
## Last updated: 24 Oct 2022

## Reference: https://marian-nmt.github.io/examples/mtm2017/complex/

model_folder="model.seq2seq.sent1";
mkdir ${model_folder};
data_path="/home/ye/exp/mysent/data-sent";
src="my"; tgt="tg";


marian \
  --type s2s \
  --train-sets ${data_path}/train.${src} ${data_path}/train.${tgt} \
  --max-length 200 \
  --valid-sets ${data_path}/valid.${src} ${data_path}/valid.${tgt} \
  --vocabs  ${data_path}/vocab/vocab.${src}.yml  ${data_path}/vocab/vocab.${tgt}.yml \
  --model ${model_folder}/model.npz \
  --workspace 4500 \
  --enc-depth 3 --enc-type alternating --enc-cell lstm --enc-cell-depth 4 \
  --dec-depth 3 --dec-cell lstm --dec-cell-base-depth 4 --dec-cell-high-depth 2 \
  --tied-embeddings --layer-normalization --skip \
  --mini-batch-fit \
  --valid-mini-batch 32 \
  --valid-metrics cross-entropy perplexity bleu\
  --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
  --dropout-rnn 0.3 --dropout-src 0.3 --exponential-smoothing \
  --early-stopping 10 \
  --log ${model_folder}/train.log --valid-log ${model_folder}/valid.log \
  --devices 0 --sync-sgd --seed 1111  \
  --dump-config > ${model_folder}/config.yml

time marian -c ${model_folder}/config.yml  2>&1 | tee ${model_folder}/s2s.${src}-${tgt}.log1
root@57452252667f:/home/ye/exp/mysent#
```

### Transformer model
```bash
root@57452252667f:/home/ye/exp/mysent# head -n 30 ./transformer.sent1.sh
#!/bin/bash

## Written by Ye Kyaw Thu, LST, NECTEC, Thailand
## Experiments for mySent, also preparation for 4th NLP/AI Workshop 2022

#     --mini-batch-fit -w 10000 --maxi-batch 1000 \
#    --mini-batch-fit -w 1000 --maxi-batch 100 \
#     --tied-embeddings-all \
#     --tied-embeddings \
#     --valid-metrics cross-entropy perplexity translation bleu \
#     --transformer-dropout 0.1 --label-smoothing 0.1 \
#     --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
#     --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \

mkdir model.transformer.sent1;

marian \
    --model model.transformer.sent1/model.npz --type transformer \
    --train-sets data-sent/train.my data-sent/train.tg \
    --max-length 200 \
    --vocabs data-sent/vocab/vocab.my.yml data-sent/vocab/vocab.tg.yml \
    --mini-batch-fit -w 1000 --maxi-batch 100 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity bleu \
    --valid-sets data-sent/valid.my data-sent/valid.tg \
    --valid-translation-output model.transformer.sent1/valid.my-tg.output --quiet-translation \
    --valid-mini-batch 32 \
    --beam-size 6 --normalize 0.6 \
    --log model.transformer.sent1/train.log --valid-log model.transformer.sent1/valid.log \
    --enc-depth 2 --dec-depth 2 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.1 \
    --learn-rate 0.0003 --lr-warmup 0 --lr-decay-inv-sqrt 16000 --lr-report \
    --clip-norm 5 \
    --tied-embeddings \
    --devices 0 --sync-sgd --seed 1111 \
    --exponential-smoothing \
    --dump-config > model.transformer.sent1/config.yml

time marian -c model.transformer.sent1/config.yml  2>&1 | tee transformer.sent1.log
root@57452252667f:/home/ye/exp/mysent#
```

## Evaluation

### WER Calculation
The SCLITE (score speech recognition system output) program from the NIST scoring toolkit (Version:2.4.11) is used to align the machine translated hypothesis tags with error-free reference tags and calculate the word error rate (WER). <br/>

Step 1: Download and Install SCTK
```
https://github.com/usnistgov/SCTK
```
Scripts: https://github.com/ye-kyaw-thu/MTRSS/tree/master/WAT2021/scripts/WER <br/>

Step 2: Add id <br/>
Directory should be ...
```
$ tree
.
├── s2s.para1.para
│   ├── hypothesis.tg
│   └── ref.tg
```
```
$ ./add-id.sh
```
wer-calc.sh is ...
```bash
#!/bin/bash

# WER calculating with sclite command
# written by Ye Kyaw Thu, LST, NECTEC, Thailand
# 5 June 2021
# $ wer-calc.sh en-my my-en

for arg in "$@"
do
   # get last 2 characters of the folder name (i.e. target language)
   trg=${arg: -2};
   cd $arg;
   for idFILE in *.id;
   do
      if [ "$idFILE" != "ref.tg.id" ]; then
         # to see some SYSTEM SUMMARY PERCENTAGES on screen 
         sclite -r ./ref.tg.id -h ./$idFILE -i spu_id
         # running with pra option 
         sclite -r ./ref.tg.id -h ./$idFILE -i spu_id -o pra
         # running with dtl option
         sclite -r ./ref.tg.id -h ./$idFILE -i spu_id -o dtl
         
      echo -e " Finished WER calculations for $fidFILE !!! \n\n"
      fi
   done
   cd ..;
done
```

### chrF Calculation

Step 1: Download chrF++ tool
```
wget https://raw.githubusercontent.com/m-popovic/chrF/master/chrF%2B%2B.py
```

Step 2: How to run ...
```
python chrf++.py -R <reference-file> -H <hypothesis-file>
```

### BLEU Calculation

Step 1: Download multi-bleu.perl

```
wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
```

Step 2: Calculate BLEU 
```
perl /home/ye/tool/multi-bleu.perl <reference-file> < <hypothesis-file> 
```
