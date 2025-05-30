import hashlib
import os
import random
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch
import time
import numpy as np
from torch.utils.data import Dataset,DataLoader
from lightning import Fabric
from tqdm import tqdm
import torch.nn.functional as F
from model.text_embedding import TextEmbeddingModel
from utils.dataset import load_datapath,SCLDataset
import argparse
from pathlib import Path
from typing import Literal
import pandas as pd

def infer(passages_dataloder,fabric,tokenizer,model,opt):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder)
        allids, allembeddings,alllabels= [],{},[]
        for layer in opt.need_layer:
            allembeddings[layer]=[]
    with torch.no_grad():
        for i,batch in enumerate(passages_dataloder):
            text,label,write_model,ids= batch
            encoded_batch = tokenizer.batch_encode_plus(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}#'input_ids':tensor,mask:tensor
            embeddings = model(encoded_batch,hidden_states=True)
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(-2),embeddings.size(-1))
            label = fabric.all_gather(write_model).view(-1)
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0:
                embeddings = F.normalize(embeddings,dim=-1).cpu().to(torch.bfloat16)
                for layer in opt.need_layer:
                    allembeddings[layer].append(embeddings[:,layer,:].clone())
                allids.extend(ids.cpu().tolist())
                alllabels.extend(label.cpu().tolist())
            del embeddings,label,ids
    if fabric.global_rank == 0:
        for layer in opt.need_layer:
            allembeddings[layer] = torch.cat(allembeddings[layer], dim=0)
        return torch.tensor(allids), allembeddings,torch.tensor(alllabels)
    else:
        return [],[],[]

def stable_long_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = (int_hash & ((1 << 63) - 1))
    return long_long_hash

def load_data(split: Literal["train", "test", "extra"], include_adversarial: bool = True, fp: str = None):
    """Load the given split of RAID into memory from the given filepath, downloading it if it does not exist.
    Returns a DataFrame.
    """
    if split not in ("train", "test", "extra"):
        raise ValueError('`split` must be one of ("train", "test", "extra")')

    fname = f"{split}.csv" if include_adversarial else f"{split}_none.csv"
    assert fp is not None, "Please provide a filepath to load the data from."
    fp = os.path.join(fp, fname)
    return pd.read_csv(fp)

class PassagesDataset(Dataset):
    def __init__(self, data):
        
        self.passages = []
        for item in data:
            if item['attack'] != 'none' and item['attack']!='paraphrase' and stable_long_hash(item['generation'])%10<5:
                continue
            self.passages.append(item)
        classes = set()
        for item in data:
            if item['model'] not in classes:
                classes.add(item['model'])
        self.classes = list(classes)
        self.human_id = self.classes.index('human')

    def __len__(self):
        return len(self.passages)
         
    def __getitem__(self, idx):
        data_now = self.passages[idx]
        text = data_now['generation']
        model = self.classes.index(data_now['model'])
        label = int(model==self.human_id)
        ids = stable_long_hash(text)
        return text, int(label),int(model), int(ids)

def test(opt):
    if opt.device_num>1:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed",devices=opt.device_num,strategy='ddp')#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed",devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name,output_hidden_states=True,lora=opt.lora,infer=True,use_pooling=opt.pooling).cuda()
    tokenizer=model.tokenizer
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
        break
    
    if 'LLM_detect_data' in opt.path:
        now_data = load_data('train',fp=opt.path)
        now_data = now_data.to_dict(orient='records')
        dataset = PassagesDataset(now_data)
        passages_dataloder = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
        passages_dataloder=fabric.setup_dataloaders(passages_dataloder)
    elif 'jsonl' not in opt.path:
        data_path = load_datapath(opt.path,include_adversarial=opt.adversarial,dataset_name=opt.database_name)['train']
        dataset = SCLDataset(data_path,fabric,tokenizer,need_ids=True,adv_p=0,has_mix=opt.has_mix)
        passages_dataloder = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
        passages_dataloder=fabric.setup_dataloaders(passages_dataloder,use_distributed_sampler=False)
    else:
        dataset = SCLDataset([opt.path],fabric,tokenizer,need_ids=True,adv_p=0)#,opt.path.replace('train','valid')
        passages_dataloder = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
        passages_dataloder=fabric.setup_dataloaders(passages_dataloder,use_distributed_sampler=False)
    model=fabric.setup(model)
    classes=dataset.classes
    train_ids, train_embeddings,train_labels = infer(passages_dataloder,fabric,tokenizer,model,opt)

    torch.cuda.empty_cache()
    if fabric.global_rank == 0:
        emb_dict={'embeddings':train_embeddings,'labels':train_labels,'ids':train_ids,'classes':classes}
        #save emb_dict to pt
        torch.save(emb_dict, f"{opt.savedir}/{opt.name}.pt")
        print(f"Save emb_dict to {opt.savedir}/{opt.name}.pt")
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--path", type=str, default="/opt/AI-text-test/UCAS/30shot/train_3.jsonl", help="Path to the dataset")
    parser.add_argument("--database_name", type=str, default="Deepfake")
    parser.add_argument("--model_path", type=str, default="ckpt/ucas_model2.pth")
    parser.add_argument('--model_name', type=str, default="FacebookAI/roberta-large")
    parser.add_argument('--lora', type=bool, default=True)
    parser.add_argument(
        "--lora_r", default=128, type=int, help="Lora r."
    )
    parser.add_argument("--lora_alpha", default=256, type=int, help="Lora alpha.")
    
    parser.add_argument('--pooling', type=str, default="max", help="Pooling method, average or cls")
    parser.add_argument('--need_layer', type=list, default=[16,17,18])
    parser.add_argument("--adversarial", type=bool, default=True)
    parser.add_argument("--has_mix", type=bool, default=False)
    
    parser.add_argument('--embedding_dim', type=int, default=1024)

    parser.add_argument("--savedir", type=str, default="runs")
    parser.add_argument("--name", type=str, default='ucas_model2_30shot_database')
    opt = parser.parse_args()
    test(opt)
