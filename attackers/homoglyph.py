import argparse
import hashlib
import json
import math
import os
import random
import regex as re
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

class HomoglyphAttack:
    def __init__(self, N=0.9):
        self.mapping = {
            "a": ["а"],
            "A": ["А", "Α"],
            "B": ["В", "Β"],
            "e": ["е"],
            "E": ["Е", "Ε"],
            "c": ["с"],
            "p": ["р"],
            "K": ["К", "Κ"],
            "O": ["О", "Ο"],
            "P": ["Р", "Ρ"],
            "M": ["М", "Μ"],
            "H": ["Н", "Η"],
            "T": ["Т", "Τ"],
            "X": ["Х", "Χ"],
            "C": ["С"],
            "y": ["у"],
            "o": ["о"],
            "x": ["х"],
            "I": ["І", "Ι"],
            "i": ["і"],
            "N": ["Ν"],
            "Z": ["Ζ"],
        }

    def attack(self, text):
        text = list(text)
        edits = []
        for i, char in enumerate(text):
            if char in self.mapping:
                text[i] = random.choice(self.mapping[char])
                edits.append((i, i + 1))

        return "".join(text)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

class HomoglyphAttacknDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attack = HomoglyphAttack()

    def __len__(self):
        return len(self.data)

    def stable_long_hash(self,input_string):
        hash_object = hashlib.sha256(input_string.encode())
        hex_digest = hash_object.hexdigest()
        int_hash = int(hex_digest, 16)
        long_long_hash = (int_hash & ((1 << 63) - 1))
        return long_long_hash

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        adv_source_id = item['id']
        text = self.attack.attack(text)
        id = self.stable_long_hash(text)
        item['text'] = text
        item['id'] = id
        item['adv_source_id'] = adv_source_id
        return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--dataset", type=str, default='Deepfake')
    args = parser.parse_args()
    data = load_jsonl(f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val.jsonl')
    dataset = HomoglyphAttacknDataset(data)
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=20,collate_fn=lambda x:x)
    out = []
    for item in tqdm(dataloder,total=len(dataloder)):
        out.append(item[0])
        # print(item[0])
        # break
    # if not os.path.exists(f'/opt/AI-text-Dataset/{args.dataset}/homoglyph/'):
    #     os.makedirs(f'/opt/AI-text-Dataset/{args.dataset}/homoglyph/')
    save_jsonl(out, f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val-attack.jsonl')