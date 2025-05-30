import pandas as pd

# 替换成你的文件路径
file_csv = '/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2.csv'
file_xlsx = '/opt/AI-text-test/UCAS/test2_prd.xlsx'

# 读取CSV文件和XLSX文件（读取名为 'predictions' 的sheet）
df_csv = pd.read_csv(file_csv)
df_xlsx = pd.read_excel(file_xlsx, sheet_name='predictions', engine='openpyxl')

# 检查行数是否一致
if len(df_csv) != len(df_xlsx):
    print(f"行数不一致：CSV 有 {len(df_csv)} 行，XLSX 有 {len(df_xlsx)} 行")
else:
    # 检查prompt字段是否一一对应
    mismatch_indices = []
    for i, (a_prompt, b_prompt) in enumerate(zip(df_csv['prompt'], df_xlsx['prompt'])):
        if str(a_prompt).strip() != str(b_prompt).strip():
            mismatch_indices.append(i)

    if not mismatch_indices:
        print("✅ 两个文件的 prompt 字段完全一一对应。")
    else:
        print(f"❌ 共发现 {len(mismatch_indices)} 处不一致的 prompt，前几个如下：")
        for i in mismatch_indices[:10]:
            print(f"行 {i}: CSV -> {df_csv['prompt'][i]!r}, XLSX -> {df_xlsx['prompt'][i]!r}")
