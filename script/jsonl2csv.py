import json
import csv

input_path = '/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2_withlabel.jsonl'   # 替换为你的 jsonl 文件路径
output_path = '/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2_withlabel.csv'       # 输出的 csv 文件名

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=['prompt', 'text', 'label'])
    writer.writeheader()

    for idx, line in enumerate(infile, 1):  # 行号从 1 开始
        obj = json.loads(line)
        # 严格要求字段存在
        if 'prompt' not in obj or 'text' not in obj or 'label' not in obj:
            raise ValueError(f"Line {idx} is missing required fields. Found: {list(obj.keys())}")
        
        writer.writerow({
            'prompt': obj['prompt'],
            'text': obj['text'],
            'label': obj['label']
        })
