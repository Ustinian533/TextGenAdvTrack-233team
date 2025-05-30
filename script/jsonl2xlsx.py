import json
import pandas as pd

def load_jsonl(file_path):
    out = []
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            out.append(item)
    print(f"Loaded {len(out)} examples from {file_path}")
    return out

# 输入输出路径
jsonl_file = '/opt/AI-text-test/UCAS/test2_prd.jsonl'
xlsx_file = '/opt/AI-text-test/UCAS/test2_prd.xlsx'

# 读取数据
data = load_jsonl(jsonl_file)

# 提取prompt和text_prediction（请将'text_prediction'替换为你真实字段名，如'score'）
df = pd.DataFrame([{
    'prompt': item['prompt'],
    'text_prediction': item.get('text_prediction', item.get('score', None))  # 自动兼容字段名
} for item in data])

# 写入Excel（带sheet名）
with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='predictions')

print(f"✅ 已保存为Excel文件：{xlsx_file}")
