# -*- coding: utf-8 -*-
"""
EDA (YoY 성장률) — 드라마 촬영지 관광 효과 분석
데이터: main_preprocessing/Final_dataset.csv
        (방문객·소비 YoY 성장률 기준 + 넷플릭스·시청률 특성 포함)
목표:   모델링 전 데이터 구조·분포·상관관계 파악 및 유효 특성 선별

실행:  python eda.py   (어느 폴더에서 실행해도 동작)
"""

import sys
import os
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

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('../main_preprocessing/Final_dataset.csv', encoding='utf-8-sig')
print(f'shape: {df.shape}  ({df.shape[0]}개 관광지-드라마 쌍 × {df.shape[1]}개 컬럼)')


# =====================================================================
# 1. 기본 정보
# =====================================================================
id_cols = ['관광지명_표준', '주분석드라마']
print('=== 분석 대상 (10개 쌍) ===')
for i, row in df[id_cols].iterrows():
    print(f'  {i+1:2d}. {row["관광지명_표준"]}  ←  {row["주분석드라마"]}')

meta_cols = ['관광지ID', '관광지명_표준', '드라마ID', '주분석드라마', '분석시군구_key']

y_cols = ['Y_visit_growth_yoy', 'Y_consume_growth_yoy']

X_cols = [c for c in df.columns if c not in meta_cols + y_cols]

print(f'메타 컬럼:  {len(meta_cols)}개')
print(f'X 특성:     {len(X_cols)}개')
print(f'Y 목표변수: {len(y_cols)}개')
print('X 특성 목록:')
for c in X_cols:
    print(f'  {c}')


# =====================================================================
# 2. Y 변수 분석 — 목표변수 분포
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = df['관광지명_표준'].tolist()

for ax, col, title, color in zip(
    axes,
    y_cols,
    ['방문객 YoY 성장률 (Y_visit_growth_yoy)', '관광소비 YoY 성장률 (Y_consume_growth_yoy)'],
    ['steelblue', 'coral']
):
    vals = df[col].values
    colors = [color if v >= 0 else 'lightgray' for v in vals]
    bars = ax.barh(labels, vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('성장률 (드라마 방영년도 vs 전년도 동기 대비)')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.003 if val >= 0 else val - 0.003,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center',
                ha='left' if val >= 0 else 'right', fontsize=8)

plt.suptitle('목표변수 (Y) 분포 — 관광지별 YoY 성장률', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_y_distribution.png', bbox_inches='tight')
plt.show()

print('\n=== Y 변수 기술통계 ===')
print(df[y_cols + ['관광지명_표준']].set_index('관광지명_표준').round(4).to_string())

# 방문객 vs 소비 성장률 산점도
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['Y_visit_growth_yoy'], df['Y_consume_growth_yoy'], s=120, zorder=5)
for _, row in df.iterrows():
    ax.annotate(row['관광지명_표준'],
                (row['Y_visit_growth_yoy'], row['Y_consume_growth_yoy']),
                textcoords='offset points', xytext=(6, 4), fontsize=8)
ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
ax.set_xlabel('방문객 YoY 성장률')
ax.set_ylabel('관광소비 YoY 성장률')
ax.set_title('방문객 vs 소비 YoY 성장률 — 두 Y변수 관계', fontweight='bold')

corr = df['Y_visit_growth_yoy'].corr(df['Y_consume_growth_yoy'])
ax.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('eda_y_scatter.png', bbox_inches='tight')
plt.show()


# =====================================================================
# 3. 결측값 분석
# =====================================================================
X_num = df[X_cols].select_dtypes(include=[np.number])

miss = X_num.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)

if len(miss) == 0:
    print('수치형 X 특성에 결측값 없음')
else:
    print(f'결측값 있는 특성: {len(miss)}개')
    print(miss.to_string())

    fig, ax = plt.subplots(figsize=(10, max(3, len(miss)*0.5)))
    miss.plot(kind='barh', ax=ax, color='tomato')
    ax.set_title('결측값 개수 (전체 10개 중)', fontweight='bold')
    ax.set_xlabel('결측 개수')
    plt.tight_layout()
    plt.savefig('eda_missing.png', bbox_inches='tight')
    plt.show()

zero_var = X_num.columns[X_num.std() == 0].tolist()
print(f'분산=0 (모든 값 동일, 제거 대상): {len(zero_var)}개')
for c in zero_var:
    print(f'  {c}: {X_num[c].unique()}')


# =====================================================================
# 4. X 특성 분포 확인
# =====================================================================
desc = X_num.describe().T
desc['cv'] = desc['std'] / (desc['mean'].abs() + 1e-9)
pd.set_option('display.max_rows', None)
print(f'=== 수치형 특성 기술통계 (전체 {len(desc)}개, CV 높은 순) ===')
print(desc.sort_values('cv', ascending=False)[['mean','std','min','max','cv']].round(3).to_string())

# 정적 특성 박스플롯
static_plot_cols = [
    '인근관광지수', '공영주차장수_1km', '시청률_평균',
    '최근접IC거리_km', '최근접공항거리_km', '최근접철도지하철역거리_km',
]
static_plot_cols = [c for c in static_plot_cols if c in X_num.columns]

fig, ax = plt.subplots(figsize=(12, 5))
X_num[static_plot_cols].boxplot(ax=ax, vert=False)
ax.set_title('주요 정적 특성 분포 (n=10)', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_static_dist.png', bbox_inches='tight')
plt.show()

# 동적 성장률 특성 분포
growth_cols = [c for c in X_num.columns if 'growth_4w' in c]

if growth_cols:
    ncols = min(len(growth_cols), 4)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    for ax, col in zip(axes, growth_cols[:4]):
        vals = X_num[col].dropna()
        idx_map = vals.index.tolist()
        ax.bar(range(len(vals)), vals.values,
               color=['steelblue' if v >= 0 else 'salmon' for v in vals.values])
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_title(col.replace('_growth_4w', '\ngrowth'), fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['관광지명_표준'], rotation=45, ha='right', fontsize=7)

    plt.suptitle('동적 특성 성장률 분포 (방송전→방송중/후 4주)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_dynamic_growth.png', bbox_inches='tight')
    plt.show()

# 넷플릭스 여부별 Y 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_col, title in zip(
    axes,
    y_cols,
    ['방문객 YoY 성장률', '관광소비 YoY 성장률']
):
    for netflix_val, label, color in [(0, '비넷플릭스', 'steelblue'), (1, '넷플릭스', 'coral')]:
        subset = df[df['넷플릭스'] == netflix_val]
        ax.bar(
            subset['관광지명_표준'],
            subset[y_col],
            label=label,
            color=color,
            alpha=0.8
        )
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_title(f'넷플릭스 여부별 {title}', fontweight='bold')
    ax.set_ylabel('YoY 성장률')
    ax.set_xticklabels(df['관광지명_표준'], rotation=30, ha='right', fontsize=8)
    ax.legend()

plt.suptitle('넷플릭스 vs 비넷플릭스 드라마 촬영지 성장률 비교', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_netflix_comparison.png', bbox_inches='tight')
plt.show()

# 시청률 vs Y 산점도
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_col, color in zip(axes, y_cols, ['steelblue', 'coral']):
    ax.scatter(df['시청률_평균'], df[y_col], s=100, color=color, zorder=5)
    for _, row in df.iterrows():
        ax.annotate(row['관광지명_표준'],
                    (row['시청률_평균'], row[y_col]),
                    textcoords='offset points', xytext=(4, 3), fontsize=8)
    r = df['시청률_평균'].corr(df[y_col])
    ax.set_title(f'시청률 vs {y_col}\nr={r:.3f}', fontweight='bold')
    ax.set_xlabel('시청률_평균')
    ax.set_ylabel(y_col)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

plt.suptitle('시청률 vs YoY 성장률', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_rating_vs_y.png', bbox_inches='tight')
plt.show()


# =====================================================================
# 5. X-Y 상관관계 분석 (특성 선별 핵심)
# =====================================================================
corr_visit = X_num.corrwith(df['Y_visit_growth_yoy']).dropna().sort_values(key=abs, ascending=False)
corr_consume = X_num.corrwith(df['Y_consume_growth_yoy']).dropna().sort_values(key=abs, ascending=False)

n_feat = max(len(corr_visit), len(corr_consume))
fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n_feat * 0.35)))

for ax, corr, title, color in zip(
    axes,
    [corr_visit, corr_consume],
    [f'Y_visit_growth_yoy 전체 {len(corr_visit)}개 상관 특성',
     f'Y_consume_growth_yoy 전체 {len(corr_consume)}개 상관 특성'],
    ['steelblue', 'coral']
):
    colors = [color if v >= 0 else 'lightgray' for v in corr.values]
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Pearson 상관계수')
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='y', labelsize=7)

plt.suptitle('X-Y 상관관계 (높을수록 유망 특성)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_xy_corr.png', bbox_inches='tight')
plt.show()

pd.set_option('display.max_rows', None)
print(f'\n=== visit YoY 상관 특성 (전체 {len(corr_visit)}개) ===')
print(corr_visit.round(3).to_string())
print(f'\n=== consume YoY 상관 특성 (전체 {len(corr_consume)}개) ===')
print(corr_consume.round(3).to_string())


# =====================================================================
# 6. X-X 상관관계 — 중복 특성 파악
# =====================================================================
ordered = list(dict.fromkeys(corr_visit.abs().index.tolist() + corr_consume.abs().index.tolist()))
top_features = ordered

corr_matrix = X_num[top_features].corr()
n = len(corr_matrix)

annot = n <= 25

fig, ax = plt.subplots(figsize=(max(14, n * 0.5), max(12, n * 0.46)))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, annot=annot, fmt='.2f', annot_kws={'size': 7},
            ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.5})
ax.set_title(f'전체 특성 간 상관관계 히트맵 ({n}개)\n(|r| > 0.8 이면 한쪽 제거 고려)', fontweight='bold')
ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('eda_feature_corr.png', bbox_inches='tight')
plt.show()

threshold = 0.85
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) >= threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], round(r, 3)))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f'|r| >= {threshold} 인 특성 쌍 ({len(high_corr_pairs)}개) — 중복 제거 대상:')
for a, b, r in high_corr_pairs:
    print(f'  r={r:+.3f}  |  {a}  ↔  {b}')


# =====================================================================
# 7. 산점도 — 전체 특성 vs Y
# =====================================================================
feats_visit = corr_visit.abs().index.tolist()

ncols = 3
nrows = int(np.ceil(len(feats_visit) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()

for ax, feat in zip(axes, feats_visit):
    x_vals = X_num[feat]
    y_vals = df['Y_visit_growth_yoy']
    valid = x_vals.notna() & y_vals.notna()
    ax.scatter(x_vals[valid], y_vals[valid], s=80, zorder=5)
    for idx in df[valid].index:
        ax.annotate(df.loc[idx, '관광지명_표준'],
                    (x_vals[idx], y_vals[idx]),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    r = x_vals[valid].corr(y_vals[valid])
    ax.set_title(f'{feat}\nr={r:.3f}', fontsize=8, fontweight='bold')
    ax.set_ylabel('Y_visit_growth_yoy', fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

for ax in axes[len(feats_visit):]:
    ax.axis('off')

plt.suptitle('전체 X 특성 vs 방문객 YoY 성장률 산점도', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_scatter_visit.png', bbox_inches='tight')
plt.show()

feats_consume = corr_consume.abs().index.tolist()

ncols = 3
nrows = int(np.ceil(len(feats_consume) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()

for ax, feat in zip(axes, feats_consume):
    x_vals = X_num[feat]
    y_vals = df['Y_consume_growth_yoy']
    valid = x_vals.notna() & y_vals.notna()
    ax.scatter(x_vals[valid], y_vals[valid], s=80, zorder=5, color='coral')
    for idx in df[valid].index:
        ax.annotate(df.loc[idx, '관광지명_표준'],
                    (x_vals[idx], y_vals[idx]),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    r = x_vals[valid].corr(y_vals[valid])
    ax.set_title(f'{feat}\nr={r:.3f}', fontsize=8, fontweight='bold')
    ax.set_ylabel('Y_consume_growth_yoy', fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

for ax in axes[len(feats_consume):]:
    ax.axis('off')

plt.suptitle('전체 X 특성 vs 관광소비 YoY 성장률 산점도', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_scatter_consume.png', bbox_inches='tight')
plt.show()


# =====================================================================
# 8. EDA 종합 — 모델링용 특성 후보 선별
# =====================================================================
min_corr = 0.30

candidate_visit = set(corr_visit[corr_visit.abs() >= min_corr].index)
candidate_consume = set(corr_consume[corr_consume.abs() >= min_corr].index)
candidates_union = candidate_visit | candidate_consume

summary = pd.DataFrame({
    'r_visit': corr_visit,
    'r_consume': corr_consume,
}).loc[list(candidates_union)]
summary['|r|_max'] = summary.abs().max(axis=1)
summary = summary.sort_values('|r|_max', ascending=False)

print(f'=== 후보 특성 (visit 또는 consume |r| >= {min_corr}) — {len(summary)}개 ===')
print(summary.round(3).to_string())

print('\n=== EDA 요약 ===')
print(f'  총 X 수치 특성:  {len(X_num.columns)}개')
print(f'  분산=0 제거:     {len(zero_var)}개')
print(f'  Y 상관 후보:     {len(candidates_union)}개 (|r| >= {min_corr})')
print(f'  고상관 쌍(중복): {len(high_corr_pairs)}쌍 (|r| >= {threshold})')
print()
print('모델링 권장사항:')
print('  1. 위 후보 특성에서 중복 제거 후 최종 선정')
print('  2. 검증 전략: Leave-One-Out CV (n=10)')
print('  3. 모델: XGBoost (max_depth=2~3) + SHAP')
print('  4. Y_visit_growth_yoy / Y_consume_growth_yoy 각각 별도 모델 학습')
