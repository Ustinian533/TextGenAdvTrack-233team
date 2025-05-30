import pandas as pd

file_csv = '/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2_withlabel.csv'
output_path = '/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2_onlylabel.csv'

df_csv = pd.read_csv(file_csv)
ground_truth = df_csv['label'].values
prompts = df_csv['prompt'].values
texts = df_csv['text'].values
print(f"Ground truth labels: {len(ground_truth)}")
print(f"Prompts: {len(prompts)}")
print(f"Texts: {len(texts)}")

labels = df_csv['label']

labels.to_csv(output_path, index=False)
