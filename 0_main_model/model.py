# -*- coding: utf-8 -*-
"""
Modeling — 드라마 촬영지 관광 효과 분석
데이터: main_preprocessing/Final_dataset.csv
모델:   XGBoost + RandomForest (비교)
검증:   Leave-One-Out CV (n=10)
해석:   SHAP (SHapley Additive exPlanations)
특성:   IC 통행량 → during 기준, 네이버 지수 → after4w/during + growth
"""

import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# =====================================================================
# 1. 데이터 로드 & 특성 선택
# =====================================================================
df = pd.read_csv('../main_preprocessing/Final_dataset.csv', encoding='utf-8-sig')

# IC 통행량: during으로 통일 (before/after 제거 — r>0.99 중복)
# 네이버_관광지지수: after4w_mean(Y 상관↑) + growth_4w, before/during 제거
# 네이버_드라마지수: during_mean + growth_4w, before/after 제거 (after↔growth r=0.93)
FEATURES = [
    '넷플릭스',
    '시청률_평균',
    '인근관광지수',
    '공영주차장수_1km',
    '최근접IC거리_km',
    '최근접공항거리_km',
    '최근접철도지하철역거리_km',
    'ic_출구통행량_during_mean',
    'ic_출구통행량_growth_4w',
    '네이버_관광지지수_after4w_mean',
    '네이버_관광지지수_growth_4w',
    '네이버_드라마지수_during_mean',
    '네이버_드라마지수_growth_4w',
]

TARGETS = ['Y_visit_growth_yoy', 'Y_consume_growth_yoy']
TARGET_KOR = {
    'Y_visit_growth_yoy':   '방문객 YoY 성장률',
    'Y_consume_growth_yoy': '관광소비 YoY 성장률',
}

X_raw = df[FEATURES].copy()
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURES)

print(f'샘플 수: {len(df)}개  |  특성 수: {len(FEATURES)}개')
miss_cols = [(c, int(X_raw[c].isnull().sum())) for c in FEATURES if X_raw[c].isnull().sum() > 0]
for col, n in miss_cols:
    print(f'  결측 처리 ({n}개 → 평균 대체): {col}')

# =====================================================================
# 2. 모델 하이퍼파라미터
# =====================================================================
XGB_PARAMS = dict(
    n_estimators=100, max_depth=2, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
RF_PARAMS = dict(
    n_estimators=500, max_depth=3,
    min_samples_leaf=1,
    random_state=42, n_jobs=-1,
)

MODEL_CONFIGS = [
    ('XGBoost',      lambda: xgb.XGBRegressor(**XGB_PARAMS)),
    ('RandomForest', lambda: RandomForestRegressor(**RF_PARAMS)),
]
MODEL_COLORS = {'XGBoost': 'steelblue', 'RandomForest': 'coral'}

# =====================================================================
# 3. LOO-CV
# =====================================================================
def run_loocv(X_arr, y_arr, model_fn):
    loo = LeaveOneOut()
    preds = np.zeros(len(y_arr))
    for train_idx, test_idx in loo.split(X_arr):
        m = model_fn()
        m.fit(X_arr[train_idx], y_arr[train_idx])
        preds[test_idx] = m.predict(X_arr[test_idx])
    return preds, mean_absolute_error(y_arr, preds), r2_score(y_arr, preds)

cv_results = {}  # (model_name, target) -> (preds, mae, r2)

for target in TARGETS:
    y = df[target].values
    for name, fn in MODEL_CONFIGS:
        preds, mae, r2 = run_loocv(X.values, y, fn)
        cv_results[(name, target)] = (preds, mae, r2)

print('\n=== LOO-CV 결과 ===')
print(f'{"모델":<14} {"타겟":<26} {"MAE":>8} {"R²":>8}')
print('-' * 60)
for (model, target), (_, mae, r2) in cv_results.items():
    print(f'{model:<14} {target:<26} {mae:>8.4f} {r2:>8.4f}')

# =====================================================================
# 4. LOO-CV 실제 vs 예측 시각화
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
site_labels = df['관광지명_표준'].tolist()

for row, target in enumerate(TARGETS):
    for col, (model_name, _) in enumerate(MODEL_CONFIGS):
        ax = axes[row][col]
        y_true = df[target].values
        y_pred, mae, r2 = cv_results[(model_name, target)]
        color = MODEL_COLORS[model_name]

        ax.scatter(y_true, y_pred, s=90, color=color, zorder=5, alpha=0.85)
        for i, name in enumerate(site_labels):
            ax.annotate(name, (y_true[i], y_pred[i]),
                        textcoords='offset points', xytext=(5, 3), fontsize=7)

        lo = min(y_true.min(), y_pred.min()) - 0.03
        hi = max(y_true.max(), y_pred.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, label='y=x')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('실제값', fontsize=9)
        ax.set_ylabel('예측값', fontsize=9)
        ax.set_title(f'{model_name} — {TARGET_KOR[target]}\nMAE={mae:.4f}  R²={r2:.4f}',
                     fontsize=10, fontweight='bold')

plt.suptitle('LOO-CV: 실제 vs 예측값 비교', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('loocv_pred_vs_true.png', bbox_inches='tight')
plt.show()
plt.close()

# =====================================================================
# 5. 전체 데이터로 최종 모델 학습 (SHAP / Feature Importance용)
# =====================================================================
final_models = {}
for target in TARGETS:
    y = df[target].values
    for name, fn in MODEL_CONFIGS:
        m = fn()
        m.fit(X.values, y)
        final_models[(target, name)] = m

# =====================================================================
# 6. Feature Importance (내장) 비교
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

for row, target in enumerate(TARGETS):
    for col, (model_name, _) in enumerate(MODEL_CONFIGS):
        ax = axes[row][col]
        m = final_models[(target, model_name)]
        imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=True)
        color = MODEL_COLORS[model_name]
        ax.barh(imp.index, imp.values, color=color, alpha=0.85)
        ax.set_title(f'{model_name} — {TARGET_KOR[target]}\n(내장 Feature Importance)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.tick_params(axis='y', labelsize=8)
        for bar, val in zip(ax.patches, imp.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7)

plt.suptitle('Feature Importance 비교 (내장 지표)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.show()
plt.close()

# =====================================================================
# 7. SHAP 분석
# =====================================================================
shap_vals = {}  # (target, model_name) -> ndarray (n_samples, n_features)

for target in TARGETS:
    for model_name, _ in MODEL_CONFIGS:
        m = final_models[(target, model_name)]
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X)
        shap_vals[(target, model_name)] = sv

        safe_t = target.replace('/', '_')
        safe_m = model_name.lower()

        # Beeswarm (dot) — 방향성 포함
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X, feature_names=FEATURES, show=False, plot_type='dot')
        plt.title(f'SHAP Beeswarm — {model_name} / {TARGET_KOR[target]}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_beeswarm_{safe_t}_{safe_m}.png', bbox_inches='tight')
        plt.show()
        plt.close()

        # Bar — mean|SHAP| 크기
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X, feature_names=FEATURES, show=False, plot_type='bar')
        plt.title(f'SHAP Bar (mean|SHAP|) — {model_name} / {TARGET_KOR[target]}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_bar_{safe_t}_{safe_m}.png', bbox_inches='tight')
        plt.show()
        plt.close()

# =====================================================================
# 8. SHAP 종합 비교 — 두 모델의 mean|SHAP| 나란히
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, target in zip(axes, TARGETS):
    rows = {}
    for model_name, _ in MODEL_CONFIGS:
        rows[model_name] = np.abs(shap_vals[(target, model_name)]).mean(axis=0)

    comp = pd.DataFrame(rows, index=FEATURES)
    comp['평균'] = comp.mean(axis=1)
    comp = comp.sort_values('평균', ascending=True)

    y_pos = np.arange(len(comp))
    w = 0.35
    ax.barh(y_pos - w/2, comp['XGBoost'],      w, label='XGBoost',      color='steelblue', alpha=0.85)
    ax.barh(y_pos + w/2, comp['RandomForest'],  w, label='RandomForest', color='coral',     alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp.index, fontsize=8)
    ax.set_xlabel('mean |SHAP|', fontsize=9)
    ax.set_title(f'SHAP 요인 중요도 비교\n{TARGET_KOR[target]}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axvline(0, color='black', lw=0.6)

plt.suptitle('XGBoost vs RandomForest — SHAP 요인 중요도 종합 비교', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_comparison.png', bbox_inches='tight')
plt.show()
plt.close()

# =====================================================================
# 9. 최종 요인 순위 출력 (두 모델 평균 mean|SHAP|)
# =====================================================================
print('\n' + '='*65)
print('요인 중요도 순위 (mean|SHAP| 기준, XGBoost & RF 평균)')
print('='*65)
ranking_dfs = {}
for target in TARGETS:
    shap_avg = (
        np.abs(shap_vals[(target, 'XGBoost')]).mean(axis=0)
        + np.abs(shap_vals[(target, 'RandomForest')]).mean(axis=0)
    ) / 2
    ranking = pd.Series(shap_avg, index=FEATURES).sort_values(ascending=False)
    ranking_dfs[target] = ranking
    print(f'\n[{TARGET_KOR[target]}]')
    for rank, (feat, val) in enumerate(ranking.items(), 1):
        xgb_val = np.abs(shap_vals[(target, 'XGBoost')]).mean(axis=0)[FEATURES.index(feat)]
        rf_val  = np.abs(shap_vals[(target, 'RandomForest')]).mean(axis=0)[FEATURES.index(feat)]
        print(f'  {rank:2d}. {feat:<40}  avg={val:.4f}  (XGB={xgb_val:.4f}, RF={rf_val:.4f})')

# =====================================================================
# 10. 최종 요인 순위 시각화 (두 Y 나란히, 평균 SHAP 기준)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, target in zip(axes, TARGETS):
    ranking = ranking_dfs[target].sort_values(ascending=True)
    color_list = ['#2196F3' if v > 0 else '#F44336' for v in ranking.values]
    bars = ax.barh(ranking.index, ranking.values, color='#5C85D6', alpha=0.9)
    ax.set_xlabel('mean |SHAP| (XGB + RF 평균)', fontsize=9)
    ax.set_title(f'최종 요인 중요도 순위\n{TARGET_KOR[target]}', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    for bar, val in zip(bars, ranking.values):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7)

plt.suptitle('드라마 촬영지 관광 효과 요인 중요도 순위 (SHAP 기반)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('factor_ranking.png', bbox_inches='tight')
plt.show()
plt.close()

# =====================================================================
# 11. 결과 CSV 저장
# =====================================================================
# LOO-CV 성능 요약
perf_rows = []
for (model, target), (_, mae, r2) in cv_results.items():
    perf_rows.append({'모델': model, '타겟': target, 'MAE': round(mae, 4), 'R²': round(r2, 4)})
pd.DataFrame(perf_rows).to_csv('loocv_summary.csv', index=False, encoding='utf-8-sig')

# 요인 순위 요약
rank_rows = []
for target in TARGETS:
    for rank, (feat, val) in enumerate(ranking_dfs[target].items(), 1):
        xgb_val = np.abs(shap_vals[(target, 'XGBoost')]).mean(axis=0)[FEATURES.index(feat)]
        rf_val  = np.abs(shap_vals[(target, 'RandomForest')]).mean(axis=0)[FEATURES.index(feat)]
        rank_rows.append({
            '타겟': TARGET_KOR[target], '순위': rank, '특성': feat,
            'mean_SHAP_avg': round(val, 4),
            'mean_SHAP_XGB': round(xgb_val, 4),
            'mean_SHAP_RF':  round(rf_val, 4),
        })
pd.DataFrame(rank_rows).to_csv('factor_ranking.csv', index=False, encoding='utf-8-sig')

print('\n저장 완료:')
print('  loocv_summary.csv   — LOO-CV 성능 결과')
print('  factor_ranking.csv  — 요인 중요도 순위')
print('  *.png               — 시각화 차트 11개')
