import json
import math
import os
import random
import torch
from torch.utils.data import Dataset

model_alias_mapping = {
    'chatgpt': 'chatgpt',
    'ChatGPT': 'chatgpt',
    'chatGPT': 'chatgpt',
    'gpt-3.5-trubo': 'gpt-3.5-trubo',
    'GPT4': 'gpt4',
    'gpt4': 'gpt4',
    'text-davinci-002': 'text-davinci-002',
    'text-davinci-003': 'text-davinci-003',
    'davinci': 'text-davinci',
    'gpt1': 'gpt1',
    'gpt2_pytorch': 'gpt2-pytorch',
    'gpt2_large': 'gpt2-large',
    'gpt2_small': 'gpt2-small',
    'gpt2_medium': 'gpt2-medium',
    'gpt2-xl': 'gpt2-xl',
    'GPT2-XL': 'gpt2-xl',
    'gpt2_xl': 'gpt2-xl',
    'gpt2': 'gpt2-xl',
    'gpt3': 'gpt3',
    'GROVER_base': 'grover_base',
    'grover_base': 'grover_base',
    'grover_large': 'grover_large',
    'grover_mega': 'grover_mega',
    'llama2-fine-tuned': 'llama2',
    'opt_125m': 'opt_125m',
    'opt_1.3b': 'opt_1.3b',
    'opt_2.7b': 'opt_2.7b',
    'opt_6.7b': 'opt_6.7b',
    'opt_13b': 'opt_13b',
    'opt_30b': 'opt_30b',
    'opt_350m': 'opt_350m',
    'opt_iml_max_1.3b': 'opt_iml_max_1.3b',
    'opt_iml_30b': 'opt_iml_30b',
    'flan_t5_small': 'flan_t5_small',
    'flan_t5_base': 'flan_t5_base',
    'flan_t5_large': 'flan_t5_large',
    'flan_t5_xl': 'flan_t5_xl',
    'flan_t5_xxl': 'flan_t5_xxl',
    'flan_t5': 'flan_t5_xxl',
    'dolly': 'dolly',
    'GLM130B': 'GLM130B',
    'bloom_7b': 'bloom_7b',
    'bloomz': 'bloomz',
    't0_3b': 't0_3b',
    't0_11b': 't0_11b',
    'gpt_neox': 'gpt_neox',
    'xlm': 'xlm',
    'xlnet_large': 'xlnet_large',
    'xlnet_base': 'xlnet_base',
    'cohere': 'cohere',
    'ctrl': 'ctrl',
    'pplm_gpt2': 'pplm_gpt2',
    'pplm_distil': 'pplm_distil',
    'fair_wmt19': 'fair_wmt19',
    'fair_wmt20': 'fair_wmt20',
    'glm130b': 'GLM130B',
    'jais-30b': 'jais',
    'transfo_xl': 'transfo_xl',
    '7B': '7B',
    '13B': '13B',
    '65B': '65B',
    '30B': '30B',
    'gpt_j': 'gpt_j',
    'mpt': 'mpt',
    'mpt-chat': 'mpt-chat',
    'llama-chat': 'llama-chat',
    'mistral': 'mistral',
    'mistral-chat': 'mistral-chat',
    'cohere-chat': 'cohere-chat',
    'human': 'human',
}

def load_datapath(path,include_adversarial=False,dataset_name='all'):
    data_path = {'train':[],'test':[]}
    if dataset_name=='all':
        datasets = os.listdir(path)
    else:
        datasets = [dataset_name]
    for dataset in datasets:
        dataset_path = os.path.join(path,dataset)
        for adv in os.listdir(dataset_path):
            if include_adversarial==False and 'no_attack' not in adv:
                continue
            adv_path = os.path.join(dataset_path,adv)
            for data in os.listdir(adv_path):
                if 'train' in data:
                    data_path['train'].append(os.path.join(adv_path,data))
                elif 'test' in data:
                    data_path['test'].append(os.path.join(adv_path,data))
                elif 'valid' in data:
                    if 'RAID' in dataset:
                        data_path['test'].append(os.path.join(adv_path,data))
                    else:
                        data_path['train'].append(os.path.join(adv_path,data))
    
    return data_path

class SCLDataset(Dataset):
    def __init__(self, data_path,fabric,tokenizer,need_ids=False,adv_p=0.5,max_length=530,name2id=None,has_mix=True):
        self.data_path = data_path
        self.adv_p = adv_p
        self.need_ids=need_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_mix = has_mix

        self.world_size = fabric.world_size
        self.global_rank = fabric.global_rank
        self.LLM_name=set()
        dataset_len = self.get_data_len(data_path)
        
        classes = sorted(list(self.LLM_name))
        if name2id is None:
            self.name2id={}
            for i,name in enumerate(classes):
                self.name2id[name]=i
        else:
            self.name2id = name2id
            for name in classes:
                assert name in self.name2id
        self.classes = classes
        print(f'there are {len(classes)} classes in dataset')
        print(f'the classes are {classes}')

        self.num_samples = math.ceil(dataset_len / self.world_size)
        total_size = self.num_samples * self.world_size
        indices = list(range(dataset_len))
        padding_size = total_size - len(indices)
        indices += indices[:padding_size]
        assert len(indices) == total_size
        indices = indices[self.global_rank : total_size : self.world_size]
        assert len(indices) == self.num_samples
        self.indices = set(indices)

        data_dict = self.load_data(data_path)
        self.dataset = [data_dict[i] for i in indices]
        self.dataset_len = len(self.dataset)

    
    def get_data_len(self,data_path):
        total_len = 0
        for path in data_path:
            print(f'reading {path}')
            with open(path, mode='r', encoding='utf-8') as jsonl_file:
                for line in jsonl_file:
                    now = json.loads(line)
                    if now['src'] not in model_alias_mapping:
                        model_alias_mapping[now['src']]=now['src']
                    now['src'] = model_alias_mapping[now['src']]
                    if self.has_mix == False:
                        if 'human' in now['src'] and now['src'] != 'human':
                            continue
                    if now['src'] not in self.LLM_name:
                        self.LLM_name.add(now['src'])
                    total_len+=1
        return total_len

    def truncate_text(self,text):
    
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    def merge_dict(self,dict1,dict2):
        for key in dict2:
            dict1[key]=dict2[key]
        return dict1

    def load_jsonl(self,file_path,total_len):
        out = {}
        cnt=0
        with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                now = json.loads(line)
                if self.has_mix == False:
                    if 'human' in now['src'] and now['src'] != 'human':
                        continue
                if total_len+cnt in self.indices:
                    out[total_len+cnt]=now
                cnt+=1
        return out,cnt

    def load_data(self,data_path):
        data = {}
        total_len = 0
        for path in data_path:
            print(f'loading {path}')
            now_data,now_len=self.load_jsonl(path,total_len)
            data = self.merge_dict(data,now_data)
            total_len+=now_len
        return data
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = data['text']
        label = data['label']
        src = self.name2id[model_alias_mapping[data['src']]]
        id = data['id']

        if self.need_ids:
            return text,int(label),int(src),int(id)
        return text,int(label),int(src)
