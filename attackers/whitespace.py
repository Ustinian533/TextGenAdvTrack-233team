import argparse
import hashlib
import json
import math
import os
import random
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

class WhiteSpaceAttack:
    def __init__(self, N=0.3):
        """
        This class takes a piece of text and randomly inserts whitespaces within the sentences.

        Args:
            N (float): Between 0 and 1, indicating the percentage of whitespaces to be altered
        """
        self.N = N

    def attack(self, text):
        # Split on spaces
        texts = text.split(" ")

        # Determine the number of spaces to insert
        spaces_to_alter = int(len(texts) * self.N)

        # Randomly sample indices to insert spaces at (with replacement!)
        indices_to_alter = random.choices(range(len(texts)), k=spaces_to_alter)

        # For all indices to alter, insert an extra space character after the token
        indices_to_alter = sorted(indices_to_alter)
        for i in indices_to_alter:
            texts[i] += " "

        return " ".join(texts)

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

class WhiteSpaceAttackDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attack = WhiteSpaceAttack()

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
    data = load_jsonl(f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val-attack1.jsonl')
    dataset = WhiteSpaceAttackDataset(data)
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=80,collate_fn=lambda x:x)
    out = []
    for item in tqdm(dataloder,total=len(dataloder)):
        out.append(item[0])
        # print(item[0])
        # break
    # if not os.path.exists(f'/opt/AI-text-Dataset/{args.dataset}/whitespace/'):
    #     os.makedirs(f'/opt/AI-text-Dataset/{args.dataset}/whitespace/')
    save_jsonl(out, f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val-attack2.jsonl')