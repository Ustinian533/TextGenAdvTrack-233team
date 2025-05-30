import csv
import json
import hashlib

def get_id(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = (int_hash & ((1 << 63) - 1))
    return long_long_hash

def save_jsonl(out,save_path):
    with open(save_path, mode='w', encoding='utf-8') as jsonl_file:
        for item in out:
            jsonl_file.write(json.dumps(item,ensure_ascii=False)+'\n')

file_path = "/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-test2.csv"  # 替换成你的CSV文件路径
save_path = "/opt/AI-text-Dataset-copy/UCAS/UCAS_AISAD_TEXT-test2.jsonl"  # 替换成你想要保存的JSONL文件路径


data = []
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:

        now_row = {}
        # print(row.keys())
        for key in row:
            if 'prompt' in key:
                now_row['prompt'] = row[key]
            elif 'text' in key:
                now_row['text'] = row[key]
            elif 'label' in key:
                now_row['label'] = int(row[key])
            else:
                raise ValueError(f"Unexpected key in CSV: {key}")
        # if now_row['label'] == 0:
        #     now_row['src'] = 'llm'
        # elif now_row['label'] == 1:
        #     now_row['src'] = 'human'
        
        now_row['id'] = get_id(now_row['text'])
        data.append(now_row)

save_jsonl(data, save_path)