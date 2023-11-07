# Introduction

# Framework
The framework shows as following picture:

# System requirement
python  3.7.11  
numpy  1.21.2  
pandas  1.1.3  
torch  1.8.0  
biopython  1.79  
sentencepiece 0.1.96  
transformers 4.15.0

# Software requirement  
To run the full & accurate version of DeepProSite, you need to make sure the following software is in the [mkdssp](./mkdssp) directory:  
[DSSP](https://github.com/cmbi/dssp) (*dssp ver 2.0.4* is Already in this repository).
The protein structures should be predicted by ESMFold to run DeepProSite:
Download the ESMFold model [guide](https://github.com/facebookresearch/esm)
You also need to prepare the pretrained language model ProtTrans:
Download the pretrained ProtT5-XL-UniRef50 model [guide](https://github.com/agemagician/ProtTrans).

# Running Commands
Inferring command:
```
CUDA_VISIBLE_DEVICES=0 python infer.py --task PRO --test --seed 2022 --run_id weights --dataset_path infer_data/PRO/ --feature_path infer_data/PRO/
```

# Pretrained Models
If you want to use our pretrained models, you can download them based on the .
