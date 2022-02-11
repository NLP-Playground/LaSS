# LaSS: Learning Language Specific Sub-network for Multilingual Machine Translation

This is the repo for ACL2021 paper Learning Language Specific Sub-network for Multilingual Machine Translation.

[paper](https://arxiv.org/abs/2105.09259)


## Introduction

LaSS, representing **La**nguage **S**pecific **S**ub-network, is a single unified multilingual MT  model. LaSS aims at alleviating the well-known parameter interference issue in multilingual MT by accommodating one sub-network for each language pair. Extensive experiments demonstrate the efficacy of LaSS and its strong generalization performance in different scenarios.


## Pre-requisite


```
pip3 install -r requirements.txt
```

## Pipeline

The pipeline contains 4 steps: 
1. Train a vanilla multilingual baseline
2. Fine-tune the baseline for each language pair
3. Obtain the masks from the fine-tuned model
4. Continue training the vanilla multilingual baseline with the obtained masks

### Data Processing

Before the training phase, you need to prepare the data. In general, data processing contains the following steps:
* Data filtering
* Data deduplication
* Learning/Applying joint BPE vocabulary
* Data Cleaning

For IWSLT we used in the paper, we directly use [this scripts](https://github.com/RayeRen/multilingual-kd-pytorch/blob/master/data/iwslt/raw/prepare-iwslt14.sh).

For WMT, we collect data from the official WMT website. For details please refer to the appendix of  our paper.

We provide some [data preprocessing scripts]() for reference.

### Multilingual baseline

We first train a vanilla multilingual baseline.

```
bash scripts/train.sh —config baseline.yml
```

### Fine-tune the baseline

After obtaining the vanilla multilingual baseline, we need to fine-tune the baseline for each language pair.

```
bash scripts/train.sh —config finetune.yml
```

After fine-tuning, we obtain n models, where n represents the number of language pairs we use.

### Obtain the masks

For each language pair, we need to prune the α percent lowest weights to obtain the sub-networks.

```
python3 toolbox/generate_mask.py —checkpoint-path xx —mask-path /path/to/destination —gen-mask-with-prob —mask-prob α —gen-part all —exclude-output-proj
```


### Training with masks

The last step is to continue training the vanilla multilingual model with the obtained masks.

```
bash scripts/train.sh —config multilingual.yml
```

The yaml config mentioned above can be found in [here]().


### Evaluation

You can evaluate the trained model with the following script:
```
bash scripts/evaluate.sh --config config.yml --checkpoint-name xxx --lang-pairs x-y --evaluate-bin /path/to/your/data
```

* `--config` is the training config. 
* `--lang-pairs` is not necessary. If not available, the script will evaluate all the language pair in the config. 
* `--evaluate-bin` is also not necessary. If not available, the script will load the data from `data_bin` in the config.


