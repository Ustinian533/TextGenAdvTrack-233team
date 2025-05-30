import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def evaluate_prediction(ground_truth_csv_path, prediction_xlsx_path):
    # 读取ground truth
    gts = pd.read_csv(ground_truth_csv_path)
    ground_truth = gts['label'].values
    print(f"Ground truth labels: {len(ground_truth)}")
    # 读取预测结果
    preds = pd.read_excel(prediction_xlsx_path, sheet_name='predictions', engine='openpyxl')
    predictions = preds['text_prediction'].values
    binary_pred = (predictions >= 0.5).astype(int)
    
    # 计算各项指标
    auc = roc_auc_score(ground_truth, predictions)
    acc = accuracy_score(ground_truth, binary_pred)
    f1 = f1_score(ground_truth, binary_pred)
    precision = precision_score(ground_truth, binary_pred)
    recall = recall_score(ground_truth, binary_pred)
    tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred).ravel()
    
    # 打印并返回结果
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fn': fn,
        'fp': fp
    }

# 示例调用方式：
result = evaluate_prediction("/opt/AI-text-test/UCAS/UCAS_AISAD_TEXT-test2_withlabel.csv", "/opt/AI-text-test/UCAS/test2_prd.xlsx")
