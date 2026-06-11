import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df = pd.read_csv('data/ald_process_data.csv')
for col in ['material', 'precursor', 'oxidant']:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col])

feature_cols = ['material_enc', 'precursor_enc', 'oxidant_enc',
                'temperature', 'pulse_time', 'purge_time']
model = xgb.XGBRegressor()
model.load_model('data/xgb_process_model.json')
importance = pd.Series(model.feature_importances_, index=feature_cols)

# Temperature window visualization
materials = ['HfO2', 'Al2O3', 'ZrO2']
colors    = ['steelblue', 'darkorange', 'seagreen']
windows   = {'HfO2': (150,300), 'Al2O3': (100,280), 'ZrO2': (150,280)}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, mat, col in zip(axes, materials, colors):
    sub = df[df['material'] == mat]
    ax.scatter(sub['temperature'], sub['gpc'], alpha=0.5, s=20, color=col)
    lo, hi = windows[mat]
    ax.axvspan(lo, hi, alpha=0.1, color='green', label='ALD window')
    ax.axvline(lo, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(hi, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('GPC (Å/cycle)')
    ax.set_title(f'{mat}'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('data/ald_temperature_window.png', dpi=150)
plt.close()

# Phase 1/2/3 comparison
model_eg = xgb.XGBRegressor(); model_eg.load_model('data/xgb_bandgap_model.json')
model_k  = xgb.XGBRegressor(); model_k.load_model('data/xgb_dielectric_model.json')
imp_eg = pd.Series(model_eg.feature_importances_, index=model_eg.get_booster().feature_names)
imp_k  = pd.Series(model_k.feature_importances_,  index=model_k.get_booster().feature_names)
print(f"Phase 1 (밴드갭) 1위: {imp_eg.idxmax()} ({imp_eg.max():.4f})")
print(f"Phase 2 (유전율) 1위: {imp_k.idxmax()} ({imp_k.max():.4f})")
print(f"Phase 3 (GPC)    1위: {importance.idxmax()} ({importance.max():.4f})")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (imp, title, col) in zip(axes, [
    (imp_eg,  'Phase 1: 밴드갭', 'steelblue'),
    (imp_k,   'Phase 2: 유전율', 'darkorange'),
    (importance, 'Phase 3: GPC', 'seagreen'),
]):
    imp.sort_values(ascending=False).head(8)[::-1].plot(kind='barh', ax=ax, color=col)
    ax.set_title(title); ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('data/ald_phase123_comparison.png', dpi=150)
plt.close()
print("저장 완료")
