import argparse
import json
import os
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from utils.index import Indexer
import torch
import time
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from lightning import Fabric
from tqdm import tqdm
from model.text_embedding import TextEmbeddingModel
import torch.nn.functional as F
from multiprocessing import Pool
from torch.nn.functional import softmax as F_softmax
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_jsonl(file_path):
    out = []
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            out.append(item)
    print(f"Loaded {len(out)} examples from {file_path}")
    return out

def gen_data(dict_data):
    embeddings = dict_data['embeddings']
    labels = dict_data['labels']
    ids = dict_data['ids']
    classes = dict_data['classes']
    # embeddings = embeddings.reshape(embeddings.shape[0],-1)
    return embeddings, labels, ids, classes

class PassagesDataset(Dataset):
    def __init__(self, data):
        
        self.passages = data

    def __len__(self):
        return len(self.passages)
         
    def __getitem__(self, idx):
        data_now = self.passages[idx]
        text = data_now['text']
        ids = data_now['id']
        return text, idx

def infer(passages_dataloder,fabric,tokenizer,model,need_layers):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder)
        allids, allembeddings= [],[]
    with torch.no_grad():
        for batch in passages_dataloder:
            text,ids= batch
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
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0:
                allembeddings.append(embeddings.cpu())
                allids.extend(ids.cpu().tolist())
    if fabric.global_rank == 0:
        allembeddings = torch.cat(allembeddings, dim=0)
        allembeddings = F.normalize(allembeddings,dim=-1)
        allembeddings = allembeddings.permute(1,0,2).numpy()
        allembeddings = {layer:allembeddings[layer] for layer in need_layers}
        return allids, allembeddings
    else:
        return [],[]

def dict2str(metrics):
    out_str=''
    if 'layer' in metrics:
        out_str+=f"layer:{metrics['layer']} "
    if 'k' in metrics:
        out_str+=f"k:{metrics['k']} "
    for key in metrics.keys():
        if key not in ['layer','k']:
            out_str+=f"{key}:{metrics[key]} "
    return out_str

def process_element(args):
    ids, scores,labels = args
    now_score = torch.zeros(2)
    sorted_indices = np.argsort(scores)[::-1]
    element_preds = {}
    
    for k, idx in enumerate(sorted_indices):
        label = labels[idx]
        now_score[label] += scores[idx]
        prob = F_softmax(now_score, dim=-1)[1].item()
        element_preds[k+1] = prob  # k+1作为key
    
    return element_preds

def save_jsonl(out,save_path):
    with open(save_path, mode='w', encoding='utf-8') as jsonl_file:
        for item in out:
            jsonl_file.write(json.dumps(item,ensure_ascii=False)+'\n')

def test(opt):

    fabric = Fabric(accelerator="cuda", devices=opt.device_num,strategy='ddp', precision="bf16-mixed")#
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name,output_hidden_states=True,lora=True,\
                               infer=True,lora_r=opt.lora_r,lora_alpha=opt.lora_alpha,use_pooling=opt.pooling).cuda()
    tokenizer=model.tokenizer
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
        break
    
    if fabric.global_rank == 0:
        all_train_embeddings, all_train_labels, all_train_ids, classes = gen_data(torch.load(opt.database_path))
        need_layers = list(all_train_embeddings.keys())
    else:
        need_layers = []
    need_layers = fabric.broadcast(need_layers)
    print(f"need_layers:{need_layers}")
    
    test_database = load_jsonl(opt.test_dataset_path)
    
    test_dataset = PassagesDataset(test_database)
    test_dataloder = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    test_dataloder=fabric.setup_dataloaders(test_dataloder)
    model=fabric.setup(model)
    st = time.time()
    test_ids, test_embeddings = infer(test_dataloder,fabric,tokenizer,model,need_layers)
    ed = time.time()
    print(f"test time:{ed-st}")
    torch.cuda.empty_cache()
    out = {}
    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        human_idx = classes.index('human')

        label_dict={}
        layer = opt.layer
        train_embeddings = all_train_embeddings[layer].float().numpy()
        if isinstance(all_train_labels,dict):
            train_labels = all_train_labels[layer].tolist()
            train_ids = all_train_ids[layer].tolist()
        else:
            train_labels = all_train_labels.tolist()
            train_ids = all_train_ids.tolist()
        
        for i in range(len(train_ids)):
            label_dict[int(train_ids[i])]=int(train_labels[i]==human_idx)
        index.label_dict = label_dict
        index.reset()
        index.index_data(train_ids, train_embeddings)
        preds={}
        for k in range(1,opt.max_K+1):
            preds[k]=[]
        now_test_embeddings=test_embeddings[layer]
        top_ids_and_scores = index.search_knn(now_test_embeddings, opt.max_K,index_batch_size=128)

        with Pool(processes=64) as pool:
            # 将数据转为可序列化格式
            args_list = [
                (ids, scores, labels)
                for ids, scores,labels in top_ids_and_scores
            ]
            
            # 按顺序获取结果（imap保持输入顺序）
            for result in tqdm(pool.imap(process_element, args_list), total=len(args_list)):
                for k, value in result.items():
                    preds[k].append(value)
        
        for idx in range(len(test_ids)):
            out[test_ids[idx]] = preds[opt.max_K][idx]
        
        print("len(out):",out[0])
        out_list = []
        for i in range(len(test_database)):
            out_list.append({'prompt':test_database[i]['prompt'],'text_prediction':out[i]})
        save_jsonl(out_list,opt.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--database_path", type=str, default="runs/ucas_model2_30shot_database.pt")
    parser.add_argument("--test_dataset_path", type=str, default="/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2.jsonl")
    parser.add_argument("--save_path", type=str, default="/opt/AI-text-test/UCAS/test2_prd.jsonl")
    parser.add_argument("--model_path", type=str, default="ckpt/ucas_model2.pth")
    parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-large')
    parser.add_argument(
        "--lora_r", default=128, type=int, help="Lora r."
    )
    parser.add_argument("--lora_alpha", default=256, type=int, help="Lora alpha.")

    parser.add_argument('--max_K', type=int, default=2, help="Search [1,K] nearest neighbors,choose the best K")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument('--pooling', type=str, default="max", help="Pooling method, average or cls")
    
    parser.add_argument('--embedding_dim', type=int, default=1024)

    parser.add_argument("--savedir", type=str, default="./runs/test")
    parser.add_argument("--name", type=str, default='cross_domains_cross_models')
    opt = parser.parse_args()
    test(opt)
