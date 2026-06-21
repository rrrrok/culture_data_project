# -*- coding: utf-8 -*-
"""
드라마 촬영지 관광 효과 발현 시점 및 지속 기간 분석
-----------------------------------------------------
데이터:
  0_ndays/Final_dataset.csv          — 드라마별 방영일자 + 회차별 시청률
  시군구별_방문객_데이터/...csv        — 일별 시군구 방문객
  일단위_지역별(시군구)_소비데이터/*.csv — 일별 지역 관광소비

분석 목표:
  1. 드라마별 방문객·소비 시계열 + 시청률 3-패널 시각화
  2. 효과 발현 시점(onset) 및 지속 기간(duration) 정량화
  3. 전체 평균 패턴 산출 및 요약 시각화

실행: python analysis.py  (어느 폴더에서 실행해도 동작)
"""

import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE = Path(__file__).parent.parent
NDAYS_PATH   = BASE / '0_ndays' / 'Final_dataset.csv'
VISIT_PATH   = BASE / '시군구별_방문객_데이터' / '시군구별_방문객_데이터.csv'
CONSUME_DIR  = BASE / '일단위_지역별(시군구)_소비데이터'
OUT_DIR      = Path(__file__).parent
PER_DRAMA_DIR = OUT_DIR / 'per_drama'
PER_DRAMA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 분석 파라미터
# ─────────────────────────────────────────────
WINDOW_BEFORE_W  = 8    # 방송 시작 전 분석 창 (주)
WINDOW_AFTER_W   = 12   # 종영 후 분석 창 (주)
BASELINE_FROM_W  = 8    # baseline 시작: 방송 시작 N주 전
BASELINE_TO_W    = 2    # baseline 종료: 방송 시작 N주 전
ONSET_THRESHOLD  = 1.05 # baseline 대비 5% 이상 상승 시 효과로 판단
MIN_CONSEC_WEEKS = 2    # 효과 인정 최소 연속 주수
ROLLING_DAYS     = 7    # 이동평균 일수

# ─────────────────────────────────────────────
# 지역 매핑 (분석시군구_key → 방문객·소비 필터)
# ─────────────────────────────────────────────
VISIT_FILTER = {
    '서울특별시 영등포구':  ('서울특별시', '영등포구'),
    '충청북도 충주시':      ('충청북도',   '충주시'),
    '서울특별시 중구':      ('서울특별시', '중구'),
    '인천광역시 중구':      ('인천광역시', '중구'),
    '부산광역시 서구':      ('부산광역시', '서구'),
    '서울특별시 종로구':    ('서울특별시', '종로구'),
    '경상북도 경주시':      ('경상북도',   '경주시'),
    '경기도 수원시 팔달구': ('경기도',     '수원시 팔달구'),
}

CONSUME_FILE = {
    '서울특별시 영등포구':  '서울특별시_영등포구_관광소비_일간집계_202201-20260528.csv',
    '충청북도 충주시':      '충주시_관광소비_일간집계_202201-20260528.csv',
    '서울특별시 중구':      '서울특별시_중구_관광소비_일간집계_202201-20260528.csv',
    '인천광역시 중구':      '인천광역시_중구_관광소비_일간집계_202201-20260528.csv',
    '부산광역시 서구':      '부산광역시_서구_관광소비_일간집계_202201-20260528.csv',
    '서울특별시 종로구':    '서울특별시_종로구_관광소비_일간집계_202201-20260528.csv',
    '경상북도 경주시':      '경상북도_경주시_관광소비_일간집계_202201-20260528.csv',
    '경기도 수원시 팔달구': '경기도_수원시_팔달구_관광소비_일간집계_202201-20260528.csv',
}

# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def parse_broadcast_dates(date_str):
    """'2024-03-09 ~ 2024-04-28' → (Timestamp, Timestamp)"""
    parts = str(date_str).strip().split('~')
    return pd.to_datetime(parts[0].strip()), pd.to_datetime(parts[1].strip())


def get_episode_dates(start, end, ep_ratings):
    """n개 시청률을 방영 기간에 균등 배분하여 날짜 시리즈 반환"""
    n = len(ep_ratings)
    if n == 0:
        return pd.Series(dtype=float), []
    dates = pd.date_range(start, end, periods=n)
    return pd.Series(ep_ratings, index=dates)


def load_visit(visit_all, sigungu_key):
    """방문객 데이터 필터링 → 날짜 인덱스 일별 시리즈"""
    sido, sgg = VISIT_FILTER[sigungu_key]
    sub = visit_all[(visit_all['시도'] == sido) & (visit_all['시군구'] == sgg)].copy()
    sub['날짜'] = pd.to_datetime(sub['날짜'])
    series = sub.set_index('날짜')['총방문객'].sort_index()
    series = series[~series.index.duplicated(keep='first')]
    return series.asfreq('D')


def load_consume(sigungu_key):
    """관광소비 데이터 로드 → 날짜 인덱스 일별 시리즈"""
    fpath = CONSUME_DIR / CONSUME_FILE[sigungu_key]
    df = pd.read_csv(fpath, encoding='utf-8-sig')
    df['날짜'] = pd.to_datetime(df['날짜'])
    series = df.set_index('날짜')['관광총소비(천원)'].sort_index()
    series = series[~series.index.duplicated(keep='first')]
    return series.asfreq('D')


def weekly_index_series(idx_series, broadcast_start, n_before=8, n_after=14):
    """
    방송 시작일 기준 상대 주(週)별 평균 인덱스 반환
    week 0 = 방송 시작 주
    """
    weeks, vals_v = [], []
    for w in range(-n_before, n_after + 1):
        w_start = broadcast_start + pd.Timedelta(weeks=w)
        w_end   = w_start + pd.Timedelta(days=6)
        seg = idx_series[w_start:w_end].dropna()
        weeks.append(w)
        vals_v.append(seg.mean() if len(seg) > 0 else np.nan)
    return pd.Series(vals_v, index=weeks)


def detect_onset_duration(weekly_idx, broadcast_end, broadcast_start):
    """
    onset_days : 방송 시작 기준 효과 발현 일수
                 음수 = 방송 전 발현
    duration_w : 종영 후 효과가 지속된 주 수
    """
    threshold = ONSET_THRESHOLD
    min_c = MIN_CONSEC_WEEKS
    end_week = int(np.ceil((broadcast_end - broadcast_start).days / 7))

    # onset 탐색: w = -4부터
    onset_week = None
    for w in weekly_idx.index:
        if w < -4:
            continue
        consec = 0
        for ww in range(w, min(w + min_c + 1, weekly_idx.index[-1] + 1)):
            if ww in weekly_idx.index and not np.isnan(weekly_idx[ww]) and weekly_idx[ww] >= threshold:
                consec += 1
            else:
                break
        if consec >= min_c:
            onset_week = w
            break

    onset_days = onset_week * 7 if onset_week is not None else np.nan

    # 종영 후 지속 기간
    duration_w = 0
    for w in range(end_week, end_week + WINDOW_AFTER_W + 1):
        if w not in weekly_idx.index or np.isnan(weekly_idx[w]):
            break
        if weekly_idx[w] >= threshold:
            duration_w += 1
        else:
            break

    peak_week = weekly_idx.idxmax() if not weekly_idx.isna().all() else np.nan
    peak_idx  = weekly_idx.max()    if not weekly_idx.isna().all() else np.nan

    return onset_days, duration_w, peak_week, peak_idx


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print('데이터 로드 중...')
ndays_df = pd.read_csv(NDAYS_PATH, encoding='utf-8-sig')
date_col = [c for c in ndays_df.columns if '방영일자' in c][0]
ep_cols  = [c for c in ndays_df.columns if '화' in c]

visit_all = pd.read_csv(VISIT_PATH, encoding='cp949', low_memory=False)
print(f'  방문객 데이터: {len(visit_all):,}행')

# ─────────────────────────────────────────────
# 드라마별 분석
# ─────────────────────────────────────────────
results = []           # 정량 결과 누적
aligned_visit   = {}   # 요약 그래프용: {label: weekly_idx_series}
aligned_consume = {}

COLOR_VISIT   = '#2196F3'
COLOR_CONSUME = '#FF7043'
COLOR_RATING  = '#9C27B0'
COLOR_BASE    = '#888888'
SHADE_COLOR   = '#BBDEFB'

for _, row in ndays_df.iterrows():
    site_name  = row['관광지명_표준']
    drama_name = row['주분석드라마']
    sg_key     = row['분석시군구_key']
    label      = f'{site_name}\n({drama_name})'
    safe_label = f"{site_name}_{drama_name}".replace(' ','_').replace('/','_')

    print(f'\n처리 중: {site_name} / {drama_name}')

    bcast_start, bcast_end = parse_broadcast_dates(row[date_col])
    bcast_weeks = (bcast_end - bcast_start).days / 7

    # 회차별 시청률
    ep_ratings = [row[c] for c in ep_cols if pd.notna(row[c])]
    ep_series  = get_episode_dates(bcast_start, bcast_end, ep_ratings)

    # 방문객·소비 로드
    v_series = load_visit(visit_all, sg_key)
    c_series = load_consume(sg_key)

    # 분석 창
    win_start = bcast_start - pd.Timedelta(weeks=WINDOW_BEFORE_W)
    win_end   = min(bcast_end + pd.Timedelta(weeks=WINDOW_AFTER_W),
                    v_series.index.max(), c_series.index.max())

    # 7일 이동평균 (centered)
    v_roll = v_series[win_start:win_end].rolling(ROLLING_DAYS, center=True, min_periods=3).mean()
    c_roll = c_series[win_start:win_end].rolling(ROLLING_DAYS, center=True, min_periods=3).mean()

    # 전년 동기 (YoY)
    yoy_start = win_start - pd.DateOffset(years=1)
    yoy_end   = win_end   - pd.DateOffset(years=1)
    v_yoy = v_series[yoy_start:yoy_end].rolling(ROLLING_DAYS, center=True, min_periods=3).mean()
    c_yoy = c_series[yoy_start:yoy_end].rolling(ROLLING_DAYS, center=True, min_periods=3).mean()
    v_yoy.index = v_yoy.index + pd.DateOffset(years=1)
    c_yoy.index = c_yoy.index + pd.DateOffset(years=1)

    # Pre-broadcast baseline
    bl_start = bcast_start - pd.Timedelta(weeks=BASELINE_FROM_W)
    bl_end   = bcast_start - pd.Timedelta(weeks=BASELINE_TO_W)
    v_base   = v_roll[bl_start:bl_end].mean()
    c_base   = c_roll[bl_start:bl_end].mean()

    v_idx = v_roll / v_base if v_base > 0 else v_roll * np.nan
    c_idx = c_roll / c_base if c_base > 0 else c_roll * np.nan

    # 주별 인덱스
    w_v = weekly_index_series(v_idx, bcast_start)
    w_c = weekly_index_series(c_idx, bcast_start)

    # Onset / Duration
    onset_v, dur_v, peak_w_v, peak_v = detect_onset_duration(w_v, bcast_end, bcast_start)
    onset_c, dur_c, peak_w_c, peak_c = detect_onset_duration(w_c, bcast_end, bcast_start)

    aligned_visit[label]   = w_v
    aligned_consume[label] = w_c

    results.append({
        '관광지명': site_name, '드라마': drama_name,
        '방영기간_주': round(bcast_weeks, 1),
        '방문객_onset_days': round(onset_v) if not np.isnan(onset_v) else '미발현',
        '방문객_peak_week': peak_w_v, '방문객_peak_index': round(peak_v, 3),
        '방문객_종영후지속_주': dur_v,
        '소비_onset_days': round(onset_c) if not np.isnan(onset_c) else '미발현',
        '소비_peak_week': peak_w_c, '소비_peak_index': round(peak_c, 3),
        '소비_종영후지속_주': dur_c,
    })

    # ── 드라마별 3-패널 그래프 ──────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    date_range = pd.date_range(win_start, win_end)

    def add_shade_and_lines(ax, v_base_val=None, c_base_val=None, is_index=True):
        ax.axvspan(bcast_start, bcast_end, alpha=0.13, color=SHADE_COLOR, label='방송 기간')
        ax.axvline(bcast_start, color='steelblue', lw=1.2, linestyle='--', alpha=0.7)
        ax.axvline(bcast_end,   color='steelblue', lw=1.2, linestyle=':',  alpha=0.7)
        if is_index:
            ax.axhline(1.0,               color=COLOR_BASE, lw=1.0, linestyle='--', alpha=0.6, label='기준선(baseline=1)')
            ax.axhline(ONSET_THRESHOLD,   color='orange',   lw=0.8, linestyle=':', alpha=0.6, label=f'효과 임계({ONSET_THRESHOLD})')

    # Panel 1: 방문객
    ax1.plot(v_roll.index, v_roll.values,       color=COLOR_VISIT, lw=2, label='방문객 (현재)')
    ax1.plot(v_yoy.index,  v_yoy.values,        color=COLOR_VISIT, lw=1.2, linestyle='--', alpha=0.45, label='방문객 (전년 동기)')
    ax1.axhline(v_base, color=COLOR_BASE, lw=1, linestyle=':', alpha=0.7, label=f'기준선 {v_base:,.0f}명')
    add_shade_and_lines(ax1, is_index=False)

    ax1r = ax1.twinx()
    ax1r.plot(v_idx.index, v_idx.values, color='navy', lw=1.2, alpha=0.5, label='정규화 지수')
    ax1r.axhline(1.0,             color='navy', lw=0.8, linestyle='--', alpha=0.4)
    ax1r.axhline(ONSET_THRESHOLD, color='orange', lw=0.8, linestyle=':', alpha=0.4)
    ax1r.set_ylabel('정규화 지수\n(현재/기준선)', fontsize=8, color='navy')
    ax1r.tick_params(axis='y', labelsize=7, colors='navy')

    if not np.isnan(onset_v):
        onset_date_v = bcast_start + pd.Timedelta(days=int(onset_v))
        ax1.axvline(onset_date_v, color='red', lw=1.5, linestyle='-.', alpha=0.8, label=f'효과 발현 ({int(onset_v):+d}일)')

    ax1.set_ylabel('총방문객 (7일 이동평균)', fontsize=9)
    ax1.set_title(f'[방문객]  발현: {int(onset_v):+d}일 | 피크 {peak_w_v}주차({peak_v:.2f}배) | 종영 후 지속 {dur_v}주'
                  if not np.isnan(onset_v) else '[방문객] 효과 미발현',
                  fontsize=9, color='steelblue', fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Panel 2: 소비
    ax2.plot(c_roll.index, c_roll.values, color=COLOR_CONSUME, lw=2, label='관광소비 (현재)')
    ax2.plot(c_yoy.index,  c_yoy.values,  color=COLOR_CONSUME, lw=1.2, linestyle='--', alpha=0.45, label='관광소비 (전년 동기)')
    ax2.axhline(c_base, color=COLOR_BASE, lw=1, linestyle=':', alpha=0.7, label=f'기준선 {c_base/1e6:.1f}백만원')
    add_shade_and_lines(ax2, is_index=False)

    ax2r = ax2.twinx()
    ax2r.plot(c_idx.index, c_idx.values, color='darkred', lw=1.2, alpha=0.5, label='정규화 지수')
    ax2r.axhline(1.0,             color='darkred', lw=0.8, linestyle='--', alpha=0.4)
    ax2r.axhline(ONSET_THRESHOLD, color='orange',  lw=0.8, linestyle=':', alpha=0.4)
    ax2r.set_ylabel('정규화 지수\n(현재/기준선)', fontsize=8, color='darkred')
    ax2r.tick_params(axis='y', labelsize=7, colors='darkred')

    if not np.isnan(onset_c):
        onset_date_c = bcast_start + pd.Timedelta(days=int(onset_c))
        ax2.axvline(onset_date_c, color='red', lw=1.5, linestyle='-.', alpha=0.8, label=f'효과 발현 ({int(onset_c):+d}일)')

    ax2.set_ylabel('관광총소비 (7일 이동평균, 천원)', fontsize=9)
    ax2.set_title(f'[관광소비]  발현: {int(onset_c):+d}일 | 피크 {peak_w_c}주차({peak_c:.2f}배) | 종영 후 지속 {dur_c}주'
                  if not np.isnan(onset_c) else '[관광소비] 효과 미발현',
                  fontsize=9, color='darkorange', fontweight='bold')
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # Panel 3: 시청률
    ax3.bar(ep_series.index, ep_series.values, width=4, color=COLOR_RATING, alpha=0.8, label='회차 시청률')
    for d, r in ep_series.items():
        ax3.text(d, r + 0.1, f'{r:.1f}%', ha='center', fontsize=6.5, color='purple')
    ax3.axvspan(bcast_start, bcast_end, alpha=0.1, color=SHADE_COLOR)
    ax3.axvline(bcast_start, color='steelblue', lw=1.0, linestyle='--', alpha=0.6)
    ax3.axvline(bcast_end,   color='steelblue', lw=1.0, linestyle=':', alpha=0.6)
    ax3.set_ylabel('시청률 (%)', fontsize=9)
    ax3.set_title(f'[시청률]  평균 {row["시청률_평균"]:.2f}%  |  {len(ep_ratings)}화', fontsize=9, color='purple', fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, max(ep_series.values) * 1.3 if len(ep_series) > 0 else 20)

    # X축 포맷
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%y.%m'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    # 방송 기간 레이블
    ax1.text(bcast_start + (bcast_end - bcast_start) / 2,
             ax1.get_ylim()[1] * 0.97,
             f'방영 기간\n({bcast_weeks:.1f}주)',
             ha='center', va='top', fontsize=8, color='steelblue',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    fig.suptitle(f'{site_name}  ←  {drama_name}\n'
                 f'방영: {bcast_start.strftime("%Y-%m-%d")} ~ {bcast_end.strftime("%Y-%m-%d")}  '
                 f'({bcast_weeks:.1f}주, {len(ep_ratings)}화)',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.savefig(PER_DRAMA_DIR / f'{safe_label}.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'  → 저장: per_drama/{safe_label}.png')
    ov_str = f'{int(onset_v):+d}일' if not np.isnan(onset_v) else 'NA'
    oc_str = f'{int(onset_c):+d}일' if not np.isnan(onset_c) else 'NA'
    print(f'     방문객  onset={ov_str}  peak={peak_v:.2f}배(@{peak_w_v}w)  dur_after_end={dur_v}w')
    print(f'     관광소비 onset={oc_str}  peak={peak_c:.2f}배(@{peak_w_c}w)  dur_after_end={dur_c}w')


# ─────────────────────────────────────────────
# 요약 1: 전체 드라마 정규화 곡선 오버레이
# ─────────────────────────────────────────────
def summary_overlay(aligned_dict, metric_name, color, bcast_weeks_list):
    weeks = range(-WINDOW_BEFORE_W, WINDOW_AFTER_W + 1)
    fig, ax = plt.subplots(figsize=(14, 6))

    all_vals = []
    for i, (lbl, wseries) in enumerate(aligned_dict.items()):
        wseries_reindexed = wseries.reindex(weeks)
        vals = wseries_reindexed.values
        all_vals.append(vals)
        alpha = 0.45
        ax.plot(weeks, vals, lw=1.3, alpha=alpha,
                label=lbl.replace('\n', '/'))

    all_arr = np.array(all_vals, dtype=float)
    mean_curve = np.nanmean(all_arr, axis=0)
    ax.plot(weeks, mean_curve, color=color, lw=3.0, zorder=10, label='▶ 전체 평균')

    ax.axhline(1.0,             color='gray',   lw=1.0, linestyle='--', alpha=0.6, label='기준선 (=1)')
    ax.axhline(ONSET_THRESHOLD, color='orange', lw=1.0, linestyle=':', alpha=0.7, label=f'효과 임계({ONSET_THRESHOLD})')
    ax.axvspan(-WINDOW_BEFORE_W, 0, alpha=0.04, color='gray')
    ax.axvline(0, color='steelblue', lw=1.5, linestyle='--', alpha=0.8, label='방송 시작(w=0)')
    ax.set_xlabel('방송 시작 기준 주(週) — 음수: 방송 전, 양수: 방송 후', fontsize=10)
    ax.set_ylabel(f'{metric_name} 정규화 지수\n(현재 / pre-broadcast 기준선)', fontsize=10)
    ax.set_title(f'드라마 방송 전후 {metric_name} 변화 패턴 — 전체 10개 드라마 비교',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=3, loc='upper left')
    ax.set_xlim(-WINDOW_BEFORE_W, WINDOW_AFTER_W)
    ax.grid(axis='y', alpha=0.3)

    safe_m = metric_name.replace(' ', '_')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'summary_overlay_{safe_m}.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'저장: summary_overlay_{safe_m}.png')
    return mean_curve


print('\n=== 요약 시각화 생성 중 ===')
bcast_weeks_list = [(pd.to_datetime(r[date_col].split('~')[1].strip()) -
                     pd.to_datetime(r[date_col].split('~')[0].strip())).days / 7
                    for _, r in ndays_df.iterrows()]

mean_v = summary_overlay(aligned_visit,   '방문객',  COLOR_VISIT,    bcast_weeks_list)
mean_c = summary_overlay(aligned_consume, '관광소비', COLOR_CONSUME, bcast_weeks_list)


# ─────────────────────────────────────────────
# 요약 2: Onset / Duration / Peak 지표 박스플롯
# ─────────────────────────────────────────────
res_df = pd.DataFrame(results)

onset_v_vals = pd.to_numeric(res_df['방문객_onset_days'],  errors='coerce').dropna()
onset_c_vals = pd.to_numeric(res_df['소비_onset_days'],    errors='coerce').dropna()
dur_v_vals   = res_df['방문객_종영후지속_주'].astype(float)
dur_c_vals   = res_df['소비_종영후지속_주'].astype(float)
peak_v_vals  = res_df['방문객_peak_index'].astype(float)
peak_c_vals  = res_df['소비_peak_index'].astype(float)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# 발현 시점
ax = axes[0]
ax.boxplot([onset_v_vals, onset_c_vals],
           patch_artist=True,
           boxprops=dict(facecolor='lightyellow'),
           medianprops=dict(color='red', lw=2))
ax.set_xticks([1, 2]); ax.set_xticklabels(['방문객', '관광소비'])
for x, vals in [(1, onset_v_vals), (2, onset_c_vals)]:
    ax.scatter([x] * len(vals), vals, color='black', s=40, zorder=5, alpha=0.7)
ax.axhline(0, color='steelblue', lw=1.2, linestyle='--', alpha=0.7, label='방송 시작일(0)')
ax.set_ylabel('방송 시작 기준 일수 (음수=사전 발현)', fontsize=9)
ax.set_title('효과 발현 시점 (onset)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
mean_ov = onset_v_vals.mean(); mean_oc = onset_c_vals.mean()
ax.text(1, ax.get_ylim()[0], f'평균\n{mean_ov:.0f}일', ha='center', fontsize=8, color='blue')
ax.text(2, ax.get_ylim()[0], f'평균\n{mean_oc:.0f}일', ha='center', fontsize=8, color='darkorange')

# 종영 후 지속 기간
ax = axes[1]
ax.boxplot([dur_v_vals, dur_c_vals],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', lw=2))
ax.set_xticks([1, 2]); ax.set_xticklabels(['방문객', '관광소비'])
for x, vals in [(1, dur_v_vals), (2, dur_c_vals)]:
    ax.scatter([x] * len(vals), vals, color='black', s=40, zorder=5, alpha=0.7)
ax.set_ylabel('종영 후 효과 지속 주수', fontsize=9)
ax.set_title('종영 후 효과 지속 기간', fontsize=11, fontweight='bold')
mean_dv = dur_v_vals.mean(); mean_dc = dur_c_vals.mean()
ax.text(1, ax.get_ylim()[0], f'평균\n{mean_dv:.1f}주', ha='center', fontsize=8, color='blue')
ax.text(2, ax.get_ylim()[0], f'평균\n{mean_dc:.1f}주', ha='center', fontsize=8, color='darkorange')

# 피크 지수
ax = axes[2]
ax.boxplot([peak_v_vals, peak_c_vals],
           patch_artist=True,
           boxprops=dict(facecolor='lightgreen'),
           medianprops=dict(color='red', lw=2))
ax.set_xticks([1, 2]); ax.set_xticklabels(['방문객', '관광소비'])
for x, vals in [(1, peak_v_vals), (2, peak_c_vals)]:
    ax.scatter([x] * len(vals), vals, color='black', s=40, zorder=5, alpha=0.7)
ax.axhline(1.0, color='gray', lw=1.0, linestyle='--', alpha=0.6, label='기준선')
ax.set_ylabel('피크 정규화 지수 (기준선 대비 최대배율)', fontsize=9)
ax.set_title('최대 효과 크기 (peak index)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
mean_pv = peak_v_vals.mean(); mean_pc = peak_c_vals.mean()
ax.text(1, ax.get_ylim()[0], f'평균\n{mean_pv:.2f}배', ha='center', fontsize=8, color='blue')
ax.text(2, ax.get_ylim()[0], f'평균\n{mean_pc:.2f}배', ha='center', fontsize=8, color='darkorange')

plt.suptitle('드라마 관광 효과 요약 지표 (전체 10개 드라마)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'summary_metrics.png', bbox_inches='tight')
plt.show()
plt.close()
print('저장: summary_metrics.png')


# ─────────────────────────────────────────────
# 요약 3: 평균 곡선 (방문객 + 소비 합산)
# ─────────────────────────────────────────────
weeks_arr = np.arange(-WINDOW_BEFORE_W, WINDOW_AFTER_W + 1)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(weeks_arr, mean_v, color=COLOR_VISIT,   lw=2.5, marker='o', ms=4, label='방문객 평균 지수')
ax.plot(weeks_arr, mean_c, color=COLOR_CONSUME, lw=2.5, marker='s', ms=4, label='관광소비 평균 지수')
ax.fill_between(weeks_arr, 1, mean_v, where=mean_v >= 1, alpha=0.12, color=COLOR_VISIT)
ax.fill_between(weeks_arr, 1, mean_c, where=mean_c >= 1, alpha=0.12, color=COLOR_CONSUME)
ax.axhline(1.0,             color='gray',   lw=1.0, linestyle='--', alpha=0.6)
ax.axhline(ONSET_THRESHOLD, color='orange', lw=0.8, linestyle=':', alpha=0.7, label=f'효과 임계({ONSET_THRESHOLD})')
ax.axvline(0, color='steelblue', lw=1.5, linestyle='--', alpha=0.8, label='방송 시작(w=0)')
ax.set_xlabel('방송 시작 기준 주(週)', fontsize=10)
ax.set_ylabel('정규화 지수 (평균)', fontsize=10)
ax.set_title('드라마 방영 전후 방문객·소비 변화 — 전체 평균 패턴', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-WINDOW_BEFORE_W, WINDOW_AFTER_W)
ax.grid(axis='y', alpha=0.3)

# 평균 onset/duration 표기
avg_ov = onset_v_vals.mean() / 7 if len(onset_v_vals) > 0 else None
avg_oc = onset_c_vals.mean() / 7 if len(onset_c_vals) > 0 else None
if avg_ov is not None:
    ax.axvline(avg_ov, color=COLOR_VISIT,   lw=1.2, linestyle='-.', alpha=0.7,
               label=f'평균 방문객 발현 ({avg_ov:.1f}주)')
if avg_oc is not None:
    ax.axvline(avg_oc, color=COLOR_CONSUME, lw=1.2, linestyle='-.', alpha=0.7,
               label=f'평균 소비 발현 ({avg_oc:.1f}주)')
ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'summary_mean_curve.png', bbox_inches='tight')
plt.show()
plt.close()
print('저장: summary_mean_curve.png')


# ─────────────────────────────────────────────
# 결과 테이블 출력 및 저장
# ─────────────────────────────────────────────
print('\n' + '='*80)
print('드라마 관광 효과 발현 시점 및 지속 기간 — 정량 결과')
print('='*80)
print(res_df.to_string(index=False))

# 전체 평균 (숫자 컬럼만)
num_cols = ['방영기간_주', '방문객_peak_index', '방문객_종영후지속_주',
            '소비_peak_index', '소비_종영후지속_주']
onset_v_num = pd.to_numeric(res_df['방문객_onset_days'], errors='coerce')
onset_c_num = pd.to_numeric(res_df['소비_onset_days'],   errors='coerce')

print('\n--- 전체 평균 ---')
for col in num_cols:
    print(f'  {col}: {res_df[col].astype(float).mean():.2f}')
print(f'  방문객_onset_days (유효 {len(onset_v_num.dropna())}개 평균): {onset_v_num.mean():.1f}일')
print(f'  소비_onset_days   (유효 {len(onset_c_num.dropna())}개 평균): {onset_c_num.mean():.1f}일')

res_df.to_csv(OUT_DIR / 'onset_duration_results.csv', index=False, encoding='utf-8-sig')
print('\n저장: onset_duration_results.csv')
print('분석 완료.')
