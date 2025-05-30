import pandas as pd
from openpyxl import load_workbook

# 现有xlsx文件路径
xlsx_path = '/opt/AI-text-test/UCAS/test2_prd.xlsx'  # <-- 替换为你的真实路径

# 要添加的统计数据
data_volume = 24000
total_time = 30.19479990005493

# 创建DataFrame
df_time = pd.DataFrame([{
    "Data Volume": data_volume,
    "Time": total_time
}])

# 载入已有工作簿并添加sheet
with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_time.to_excel(writer, index=False, sheet_name='time')

print("✅ 已成功添加 'time' 工作表")
