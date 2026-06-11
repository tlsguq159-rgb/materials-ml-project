import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

df = pd.read_csv('data/ald_process_data.csv')
cat_cols = ['material', 'precursor', 'oxidant']
for col in cat_cols:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col])

feature_cols = ['material_enc', 'precursor_enc', 'oxidant_enc',
                'temperature', 'pulse_time', 'purge_time']
X, y = df[feature_cols], df['gpc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbosity=0)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}  R²: {r2:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue')
lims = [min(y_test.min(), y_pred.min())-0.05, max(y_test.max(), y_pred.max())+0.05]
axes[0].plot(lims, lims, 'r--', linewidth=1)
axes[0].set_xlabel('Actual GPC'); axes[0].set_ylabel('Predicted GPC')
axes[0].set_title(f'MAE={mae:.3f}  R²={r2:.3f}')
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance.sort_values()[::-1].plot(kind='barh', ax=axes[1], color='seagreen')
axes[1].set_title('Feature Importance')
plt.tight_layout()
plt.savefig('data/process_model_eval.png', dpi=150)
plt.close()
model.save_model('data/xgb_process_model.json')
print("저장 완료")
