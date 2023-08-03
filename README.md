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
CUDA_VISIBLE_DEVICES=0 python infer_lms.py --task ZN --test --seed 2022 --run_id run_name0409_PCA_main10407_6 --dataset_path infer_data/ZN/ --feature_path infer_data/ZN/
```

# Pretrained Models

If you want to use our pretrained models, you can download them based on the [url](https://github.com/lhotse-speech/lhotse/pull/1072/files).
