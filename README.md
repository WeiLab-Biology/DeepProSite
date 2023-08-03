# Introduction
MullBind is an accurate predic

# Framework
The framework shows as following picture:
![](imgs/ucas_logo.jpg)

# Environment
python==3.8

cuda==11.2


# Dataset
We use ZN as our training dataset. 

# Feature Extract
## extract graph
## extract t5
## fuse graph and t5

# Running Commands
Inferring command:

```
CUDA_VISIBLE_DEVICES=0 python infer.py --task PRO --test --seed 2022 --run_id weights --dataset_path infer_data/PRO/ --feature_path infer_data/PRO/
```

# Pretrained Models

If you want to use our pretrained models, you can download them based on the [url](https://github.com/lhotse-speech/lhotse/pull/1072/files).
