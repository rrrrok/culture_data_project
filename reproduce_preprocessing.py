"""K-drama tourism factor-analysis preprocessing reproduction script.

This script rebuilds the final 10-event modeling dataset from the approved
intermediate workbooks. It intentionally does not use the discarded
available-filled Y calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY = ["관광지ID", "드라마ID"]
MIN_MONTH_DAYS = 28

FINAL_COLUMNS = [
    "Y_visit_growth_yoy",
    "Y_consume_growth_yoy",
    "관광지ID",
    "관광지명_표준",
    "드라마ID",
    "주분석드라마",
    "분석시군구_key",
    "넷플릭스",
    "시청률_평균",
    "인근관광지수",
    "공영주차장수_1km",
    "최근접IC거리_km",
    "최근접공항거리_km",
    "최근접철도지하철역거리_km",
    "ic_출구통행량_before4w_mean",
    "ic_출구통행량_during_mean",
    "ic_출구통행량_after4w_mean",
    "ic_출구통행량_growth_4w",
    "네이버_관광지지수_before4w_mean",
    "네이버_관광지지수_during_mean",
    "네이버_관광지지수_after4w_mean",
    "네이버_관광지지수_growth_4w",
    "네이버_드라마지수_before4w_mean",
    "네이버_드라마지수_during_mean",
    "네이버_드라마지수_after4w_mean",
    "네이버_드라마지수_growth_4w",
]

STATIC_COLUMNS = [
    "관광지ID",
    "관광지명_표준",
    "드라마ID",
    "주분석드라마",
    "분석시군구_key",
    "넷플릭스",
    "시청률_평균",
    "인근관광지수",
    "공영주차장수_1km",
    "최근접IC거리_km",
    "최근접공항거리_km",
    "최근접철도지하철역거리_km",
]

DYNAMIC_COLUMNS = [
    "관광지ID",
    "드라마ID",
    "ic_출구통행량_before4w_mean",
    "ic_출구통행량_during_mean",
    "ic_출구통행량_after4w_mean",
    "ic_출구통행량_growth_4w",
    "네이버_관광지지수_before4w_mean",
    "네이버_관광지지수_during_mean",
    "네이버_관광지지수_after4w_mean",
    "네이버_관광지지수_growth_4w",
    "네이버_드라마지수_before4w_mean",
    "네이버_드라마지수_during_mean",
    "네이버_드라마지수_after4w_mean",
    "네이버_드라마지수_growth_4w",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="최종 요인분석용 데이터셋을 재현합니다."
    )
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--y-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-final",
        type=Path,
        help="기존 최종 파일과 값이 같은지 검증할 때 사용합니다.",
    )
    return parser.parse_args()


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in KEY:
        result[column] = result[column].astype("string").str.strip()
    return result


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name}에 필요한 컬럼이 없습니다: {missing}")


def validate_unique_key(df: pd.DataFrame, name: str) -> None:
    missing_keys = int(df[KEY].isna().any(axis=1).sum())
    duplicate_keys = int(df.duplicated(KEY, keep=False).sum())
    if missing_keys or duplicate_keys:
        raise ValueError(
            f"{name} 키 검증 실패: 결측 키 {missing_keys}건, "
            f"중복 키 행 {duplicate_keys}건"
        )


def safe_growth(current: float, previous: float):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return pd.NA
    return (current - previous) / previous


def exclusion_reason(row: pd.Series) -> str:
    reasons = []
    if row["대상월사용일수"] < MIN_MONTH_DAYS:
        reasons.append("대상월 관측일수 28일 미만")
    if row["전년월사용일수"] < MIN_MONTH_DAYS:
        reasons.append("전년동월 관측일수 28일 미만")
    return " / ".join(reasons)


def recalculate_y(
    month_log: pd.DataFrame, data_name: str, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate YoY Y after excluding incomplete month pairs."""
    required = [
        *KEY,
        "데이터",
        "대상월",
        "대상월합계",
        "전년동월합계",
        "대상월사용일수",
        "전년월사용일수",
    ]
    require_columns(month_log, required, "Y 월별_검증 시트")
    subset = normalize_keys(month_log.loc[month_log["데이터"].eq(data_name)])

    summaries = []
    details = []
    for (attraction_id, drama_id), group in subset.groupby(KEY, sort=False):
        group = group.copy()
        group["집계포함여부"] = (
            group["대상월사용일수"].ge(MIN_MONTH_DAYS)
            & group["전년월사용일수"].ge(MIN_MONTH_DAYS)
        ).astype(int)
        group["제외사유"] = group.apply(
            lambda row: ""
            if row["집계포함여부"] == 1
            else exclusion_reason(row),
            axis=1,
        )

        included = group.loc[group["집계포함여부"].eq(1)]
        excluded = group.loc[group["집계포함여부"].eq(0)]
        target_sum = included["대상월합계"].sum(min_count=1)
        previous_sum = included["전년동월합계"].sum(min_count=1)

        summaries.append(
            {
                "관광지ID": attraction_id,
                "드라마ID": drama_id,
                f"{prefix}_target_sum": target_sum,
                f"{prefix}_previous_sum": previous_sum,
                f"Y_{prefix}_growth_yoy": safe_growth(target_sum, previous_sum),
                f"{prefix}_used_month_count": len(included),
                f"{prefix}_excluded_months": ", ".join(
                    excluded["대상월"].astype(str)
                ),
            }
        )
        details.append(group.assign(재계산대상=prefix))

    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def build_y(y_source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_log = pd.read_excel(y_source, sheet_name="월별_검증")
    visit, visit_log = recalculate_y(month_log, "총방문객", "visit")
    consume, consume_log = recalculate_y(month_log, "관광총소비", "consume")
    y = visit.merge(consume, on=KEY, how="inner", validate="one_to_one")
    validate_unique_key(y, "재계산 Y")
    return y, pd.concat([visit_log, consume_log], ignore_index=True)


def build_final(
    static_path: Path, dynamic_path: Path, y_source: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = normalize_keys(pd.read_excel(static_path, sheet_name="X_static"))
    dynamic = normalize_keys(
        pd.read_excel(dynamic_path, sheet_name="X_dynamic_이벤트집계")
    )
    require_columns(static, STATIC_COLUMNS, "X_static")
    require_columns(dynamic, DYNAMIC_COLUMNS, "X_dynamic")
    validate_unique_key(static, "X_static")
    validate_unique_key(dynamic, "X_dynamic")

    y, month_log = build_y(y_source)
    final = (
        static[STATIC_COLUMNS]
        .merge(dynamic[DYNAMIC_COLUMNS], on=KEY, how="inner", validate="one_to_one")
        .merge(
            y[KEY + ["Y_visit_growth_yoy", "Y_consume_growth_yoy"]],
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
    )
    final = final[FINAL_COLUMNS].sort_values(KEY).reset_index(drop=True)
    validate_unique_key(final, "Final_dataset")
    return final, month_log


def compare_with_expected(
    actual: pd.DataFrame, expected_path: Path | None
) -> dict[str, object]:
    if expected_path is None:
        return {"expected_file_checked": False}

    expected = normalize_keys(
        pd.read_excel(expected_path, sheet_name="Final_dataset")
    )
    require_columns(expected, FINAL_COLUMNS, "기존 최종 데이터")
    expected = expected[FINAL_COLUMNS].sort_values(KEY).reset_index(drop=True)

    differences = []
    for column in FINAL_COLUMNS:
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            left_num = pd.to_numeric(left, errors="coerce")
            right_num = pd.to_numeric(right, errors="coerce")
            equal = (left_num - right_num).abs().fillna(0).le(1e-10)
            equal |= left_num.isna() & right_num.isna()
        else:
            equal = left.fillna("<NA>").astype(str).eq(
                right.fillna("<NA>").astype(str)
            )
        if not bool(equal.all()):
            differences.append(
                {"column": column, "different_rows": int((~equal).sum())}
            )

    return {
        "expected_file_checked": True,
        "expected_rows": len(expected),
        "expected_columns": len(expected.columns),
        "matches_expected": not differences,
        "differences": differences,
    }


def save_outputs(
    final: pd.DataFrame,
    month_log: pd.DataFrame,
    output_dir: Path,
    comparison: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Final_dataset_reproduced.csv"
    xlsx_path = output_dir / "Final_dataset_reproduced.xlsx"
    qa_path = output_dir / "reproduction_qa.json"

    excluded = month_log.loc[month_log["집계포함여부"].eq(0)].copy()
    qa = {
        "rows": len(final),
        "columns": len(final.columns),
        "duplicate_key_rows": int(final.duplicated(KEY, keep=False).sum()),
        "Y_visit_missing": int(final["Y_visit_growth_yoy"].isna().sum()),
        "Y_consume_missing": int(final["Y_consume_growth_yoy"].isna().sum()),
        "excluded_month_pair_rows": len(excluded),
        **comparison,
    }

    final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="Final_dataset", index=False)
        month_log.to_excel(writer, sheet_name="Y_월별계산로그", index=False)
        excluded.to_excel(writer, sheet_name="Y_제외월", index=False)
        pd.DataFrame(
            [{"점검항목": key, "결과": json.dumps(value, ensure_ascii=False)}
             for key, value in qa.items()]
        ).to_excel(writer, sheet_name="검증", index=False)

    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return qa


def main() -> None:
    args = parse_args()
    final, month_log = build_final(args.static, args.dynamic, args.y_source)
    comparison = compare_with_expected(final, args.expected_final)
    qa = save_outputs(final, month_log, args.output_dir, comparison)
    print(json.dumps(qa, ensure_ascii=False, indent=2))

    if len(final) != 10:
        raise ValueError(f"최종 행 수가 예상 10행과 다릅니다: {len(final)}행")
    if qa["duplicate_key_rows"] != 0:
        raise ValueError("최종 데이터에 관광지ID+드라마ID 중복이 있습니다.")
    if qa["Y_visit_missing"] != 0 or qa["Y_consume_missing"] != 0:
        raise ValueError("최종 Y에 결측값이 있습니다.")
    if comparison.get("expected_file_checked") and not comparison.get(
        "matches_expected"
    ):
        raise ValueError("재현 결과가 기존 최종 파일과 일치하지 않습니다.")


if __name__ == "__main__":
    main()
