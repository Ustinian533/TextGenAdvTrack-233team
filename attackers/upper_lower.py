import nltk
import argparse
import hashlib
import json
import math
import os
import random
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm


class UpperLowerFlipAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and flips the case of the first letter of words based on N%.
        Most effective on articles.

        Args:
            N (float): Between 0 and 1, indicating the percentage of first letters of words to be altered
        """
        self.N = N
        self.tokenizer = nltk.tokenize.NLTKWordTokenizer()

    def attack(self, text):
        # Get all indices of starts of token spans where the first char is alphabetic
        indices = [s for s, e in self.tokenizer.span_tokenize(text) if text[s].isalpha()]

        # Get the number of indices to be changed based on the percentage N
        num_to_flip = math.ceil(len(indices) * self.N)

        # Select num_to_flip indices randomly from the candidate indices
        flip_indices = random.sample(indices, num_to_flip)

        # Flip indices to upper if lower and to lower if upper
        text = list(text)  # Cast to list since python strings are immutable
        for i in flip_indices:
            text[i] = text[i].lower() if text[i].isupper() else text[i].upper()

        # Get the spans for the flipped indices to be consistent with the format
        edits = [(i, i + 1) for i in flip_indices]

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

class UpperLowerFlipDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attack = UpperLowerFlipAttack()

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
    data = load_jsonl(f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val-attack.jsonl')
    dataset = UpperLowerFlipDataset(data)
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=80,collate_fn=lambda x:x)
    out = []
    for item in tqdm(dataloder,total=len(dataloder)):
        out.append(item[0])
        # print(item[0])
        # break
    # if not os.path.exists(f'/opt/AI-text-Dataset/{args.dataset}/upper_lower/'):
    #     os.makedirs(f'/opt/AI-text-Dataset/{args.dataset}/upper_lower/')
    save_jsonl(out, f'/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-val-attack1.jsonl')

