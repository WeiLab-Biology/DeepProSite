# Introduction
DeepProSite is a framework for identifying protein binding sites that utilizes protein structure and sequence information. 
DeepProSite is fast and accurate, and it's easy to install and run. The DeepProSite web server is freely available in [here](https://inner.wei-group.net/DeepProSite).

# System requirement
python  3.7.11  
numpy  1.21.2  
pandas  1.1.3  
torch  1.8.0  
biopython  1.79  
sentencepiece 0.1.96  
transformers 4.15.0

# Install and set up DeepProSite
**1.** Clone this repository by `git clonehttps://github.com/WeiLab-Biology/DeepProSite.git` or download the code in ZIP archive.  我们只提供了蛋白质的例子。
**2.** Download the [ESMFold](https://github.com/facebookresearch/esm) model and install according to the official tutorials：
```
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```  
Predicts structures via `python scripts/fold.py -i example/test_protein.fa -o example/pdb --cpu-offload`
版权?


**3.** Download the pre-trained ProtT5-XL-UniRef50 model in [here](https://github.com/agemagician/ProtTrans). The downloaded model is stored in ` pretrained_model/Rostlab/prot_t5_xl_uniref50`.  
Extracting language model features and normalize them based on the training set：
```
python script/feature_extraction/ProtTrans.py --fasta example/test_protein.fa --out_path example/rawembedd --gpu 0
python script/feature_extraction/process_ProtTrans.py
```

**4.** Add permission to execute for [DSSP](https://github.com/cmbi/dssp)  and extract dssp:
```
python script/feature_extraction/get_dssp.py 
```
**5.** Merge features:
```
python script/feature_extraction/pad_feature.py 
```

# Run DeepProSite for prediction
Run the following command to predict the binding sites of the sequence in "example.fa" on GPU (id=0):
```
CUDA_VISIBLE_DEVICES=0 python infer.py --task PRO --test --seed 2022 --run_id weights --dataset_path infer_data/PRO/ --feature_path infer_data/PRO/
```

# Datasets and models
The datasets used in this study are stored in `./datasets/`  
The trained DeepProSite models can be found in `./model/`
