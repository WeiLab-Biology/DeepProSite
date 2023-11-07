###YQM
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type = str, default = None)
parser.add_argument("--out_path", type = str, default = None)
parser.add_argument("--gpu", type = str, default = None)
args = parser.parse_args()

fasta_file = args.fasta
output_path = args.out_path
gpu = args.gpu

ID_list = []
seq_list = []
with open(fasta_file, "r") as f:
    lines = f.readlines()
for line in lines:
    if line[0] == ">":
        ID_list.append(line[1:-1])
    else:
        seq_list.append(" ".join(list(line.strip())))


model_path = "./pretrained_model/Rostlab/prot_t5_xl_uniref50"
# Load the vocabulary and ProtT5-XL-UniRef50 Model
tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_path)
gc.collect()

# Load the model into the GPU if avilabile and switch to inference mode
device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
model = model.to(device)
model = model.eval()


batch_size = 10

for i in tqdm(range(0, len(ID_list), batch_size)):
    if i + batch_size <= len(ID_list):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]
    else:
        batch_ID_list = ID_list[i:]
        batch_seq_list = seq_list[i:]

    # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
    batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

    # Tokenize, encode sequences and load it into the GPU if possibile
    ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Extracting sequences' features and load it into the CPU if needed
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        np.save(output_path + "/" + batch_ID_list[seq_num], seq_emd)
