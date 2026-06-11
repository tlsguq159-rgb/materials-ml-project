# scripts/run_eda_process.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/ald_process_data.csv')
print(f"데이터: {len(df)}행")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. GPC 분포 (소재별)
for mat, grp in df.groupby('material'):
    axes[0,0].hist(grp['gpc'], bins=15, alpha=0.6, label=mat)
axes[0,0].set_xlabel('GPC (Å/cycle)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('GPC Distribution by Material')
axes[0,0].legend()

# 2. Temperature vs GPC
colors = {'HfO2': 'steelblue', 'Al2O3': 'darkorange', 'ZrO2': 'seagreen'}
for mat, grp in df.groupby('material'):
    axes[0,1].scatter(grp['temperature'], grp['gpc'],
                      alpha=0.6, s=20, color=colors[mat], label=mat)
axes[0,1].set_xlabel('Temperature (°C)')
axes[0,1].set_ylabel('GPC (Å/cycle)')
axes[0,1].set_title('Temperature vs GPC — ALD Window 확인')
axes[0,1].legend()

# 3. Pulse time vs GPC
axes[1,0].scatter(df['pulse_time'], df['gpc'], alpha=0.5, s=20, color='purple')
axes[1,0].set_xlabel('Pulse Time (s)')
axes[1,0].set_ylabel('GPC (Å/cycle)')
axes[1,0].set_title('Pulse Time vs GPC — Saturation 확인')

# 4. Precursor 분포
df['precursor'].value_counts().plot(kind='bar', ax=axes[1,1], color='steelblue')
axes[1,1].set_xlabel('Precursor')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Precursor Distribution')
axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('data/ald_eda.png', dpi=150)
plt.close()
print("저장 완료: data/ald_eda.png")

print(f"\n=== GPC 통계 ===")
print(df.groupby('material')['gpc'].describe().round(3))
suspicious = df[df['gpc'] > 4.0]
print(f"\nGPC > 4.0 (CVD 모드 의심): {len(suspicious)}개")
