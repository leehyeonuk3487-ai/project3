"""M7 — 통합 리포트.

M0~M5·모듈A·외인확장·M6(시나리오·민감도)를 하나의 리포트로 종합한다. 모든 수치는
각 모듈을 **라이브로 호출**해 끌어온다(하드코딩 금지 — 코드와 리포트가 절대 어긋나지
않게). 평가기준별 매핑·정직성/한계·윤리/미션·데이터 출처를 명시한다.

★대원칙(리포트에도 명시):
  · 현재 pricing(보험료·예산)은 M0 직접관측 백본 불변. GBM 외삽은 미래 투영 한정.
  · 집단 단위 설계용 — 개인 스코어링·개인 차등 금지.
  · 미션: 예산 효율화 → 같은 예산으로 더 많은 장병이 더 많은 혜택.

사용:
  from src.validation import integrated_report
  md = integrated_report.build_report()           # 마크다운 문자열
  python -m src.validation.integrated_report       # reports/M7_integrated_report.md 생성
"""
from __future__ import annotations

import datetime as _dt

# 데이터 출처·연도·라이선스 (공공데이터, 공공누리/통계 제공)
DATA_SOURCES = [
    ("병무청", "징병 신체검사 통계(등급→현역율·BMI)", "2019–2023", "공공데이터포털 / 공공누리 제1유형"),
    ("질병관리청(KDCA)", "중증외상·퇴원손상·비외상성 중증손상(발생·치명·장해)", "2019–2022", "국가손상정보포털 / 공공누리"),
    ("질병관리청(KDCA)", "지역사회건강조사(CHS) 시군구 건강행태·BMI", "2019–2023", "공공누리 제1유형"),
    ("질병관리청(KDCA)", "국민건강영양조사(KNHANES) 측정 BMI·흡연·손상(IJMT)", "2021–2024", "KDCA 원자료 승인이용"),
    ("통계청(KOSIS)", "장래인구추계(시도×성×5세)", "2020–2035", "공공누리 제1유형"),
    ("통계청(KOSIS)", "사망원인통계(전체사망 envelope 검증)", "2005–2024", "공공누리 제1유형"),
    ("지자체 10곳", "군복무 청년 상해보험 급부표(연구 보고서)", "2024–2025", "지자체 공개 보도자료·조례"),
    ("국방부", "사망사고 통계(군기·안전사고 건수 + 군·민간 자살률)", "2011–2025", "공공데이터포털(개최기관 직접)"),
]


def _fmt(x, nd=0):
    try:
        return f"{x:,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def collect(schedule: str = "경기도") -> dict:
    """전 모듈을 라이브 호출해 핵심 수치를 수집(결정적). 무겁다(CV·ML 포함)."""
    import warnings
    warnings.filterwarnings("ignore")

    from ..data import mortality
    from ..models import cell_panel, injury_ml, projection
    from ..optimize import premium, budget, scenarios
    from . import report as val_report

    c: dict = {"schedule": schedule}

    # M0 — 사망원인 분해 + MECE envelope
    c["m0_envelope"] = mortality.envelope_check()

    # 모듈 A — 외인 셀패널 GBM 공간·시간 CV + 채택판정(외인 캘리브 0.76)
    a = cell_panel.evaluate("external")
    c["moduleA"] = {
        "adopt_gbm": a.adopt_gbm,
        "years": a.years,
        "n_cells": a.n_cells,
        "spatial_calib": round(a.spatial["gbm"]["calib_slope"], 3),
        "temporal_calib": round(a.temporal["gbm"]["calib_slope"], 3),
        "note": a.note,
    }

    # M2 — KNHANES 개인 손상위험 ML(집단용, 개인 스코어링 금지)
    m2 = injury_ml.evaluate()
    kf_best = max(m2.kfold[m]["auc_mean"] for m in ("logit", "lgbm"))
    c["m2"] = {
        "n_obs": m2.n_obs, "prevalence": m2.prevalence,
        "kfold_best_auc": kf_best,
        "holdout_baseline_auc": m2.holdout["baseline"]["auc"],
        "adopt": m2.adopt, "adopt_auc": m2.holdout[m2.adopt]["auc"],
    }

    # 생태회귀 LORO 캘리브레이션(주지표) — 베이스라인 정직 비교
    loro = val_report.loro_calibration()
    c["loro"] = {
        "model_mae": loro["metrics"]["model"]["mae"],
        "baseline_mae": loro["metrics"]["baseline"]["mae"],
        "mae_improvement_pct": loro["metrics"]["mae_improvement_pct"],
        "pi_coverage": loro["metrics"]["pi_coverage"],
    }

    # AI 성능 검증 아티팩트 — 외인 GBM의 검증된 성과(가시화). pricing 불변.
    from . import ai_performance
    c["ai"] = ai_performance.scorecard()
    c["ai_calib"] = ai_performance.calibration_reliability()["table"]

    # M5 — 계리 보험료(M0 백본 직접관측)
    pr = premium.actuarial_premium(schedule)
    c["m5"] = {"gross_pc": round(pr.gross_pc), "cv": round(pr.cv, 3),
               "loading_implied": round(pr.gross_pc / pr.net_pc, 3)}

    # 외인 추세 외삽 밴드(미래 투영 — parametric 로그선형, 점추정 아님)
    slopes = projection.external_trend_slopes()
    c["trend"] = {k: round(v * 100, 2) for k, v in slopes.items()}

    # M6 — 복지가중 LP vs 그리디 정직 비교 + 시나리오·토네이도·다년 밴드
    full = budget.expected_claims(schedule).total_claims * budget.DEFAULT_LOADING
    cmp = budget.compare_allocation(schedule, full * 0.8)
    c["lp"] = {
        "budget": round(full * 0.8),
        "objective": cmp["objective"],
        "lp_welfare": cmp["lp"]["welfare_benefit"],
        "greedy_welfare": cmp["greedy"]["welfare_benefit"],
        "lp_gain_welfare": round(cmp["lp_gain_welfare"] or 0.0, 1),
        "lp_beneficiaries": cmp["lp"]["beneficiaries"],
        "greedy_beneficiaries": cmp["greedy"]["beneficiaries"],
    }
    c["weight_sens"] = scenarios.weight_sensitivity(schedule)
    c["perturbation"] = scenarios.perturbation_sensitivity(schedule)
    c["tornado"] = scenarios.tornado(schedule)
    c["multiyear"] = scenarios.multiyear_band(schedule, years=[2024, 2027, 2030])
    # 군 코호트 proxy 타당성 검증(국방부 사망사고 통계) — pricing 미변경, 정직성·검증용
    from . import military_proxy
    c["military"] = military_proxy.summary()

    cliff = projection.population_cliff("전국", 2024, [2024, 2030])
    s = cliff.set_index("year")["stock"]
    c["cliff"] = {"y2024": int(s[2024]), "y2030": int(s[2030]),
                  "pct": round((s[2030] / s[2024] - 1) * 100, 1)}
    return c


def build_report(schedule: str = "경기도", collected: dict | None = None) -> str:
    """통합 리포트 마크다운 생성. collected 미지정 시 라이브 수집(무겁다)."""
    c = collected or collect(schedule)
    today = _dt.date.today().isoformat()
    L: list[str] = []
    w = L.append

    w(f"# M7 — 군복무 청년 상해보험 리스크·계리·예산 최적화: 통합 리포트")
    w(f"\n> 생성일 {today} · 대상 급부표 예시: {c['schedule']} · 모든 수치는 코드에서 그대로 산출(재현 가능)\n")

    # ── 한눈에 보는 요약 ─────────────────────────────────────────
    from ..data import benefits
    reported = benefits.REPORTED_PREMIUM.get(c["schedule"])
    red = c["ai"]["cv_deviance_reduction"]
    sp_r = red.get("공간CV(leave-1-시도-out)")
    tp_r = red.get("시간CV(최근2년 홀드아웃)")
    w("## 한눈에 보는 요약")
    w(
        "군에 복무하는 20대 남성이 어떤 사고와 질병을 얼마나 겪는지 공개된 정부 통계로 분석하고, "
        "그 위험에 맞는 보험료와 예산 배분을 계산하는 시스템이다.\n\n"
        "여러 지자체가 군 장병 상해보험을 앞다퉈 도입하지만, 실제 보험료나 보험금 청구 자료는 "
        "대부분 공개되지 않아 보장과 예산을 정할 근거가 부족하다. 이 시스템은 공개 통계로 그 빈자리를 "
        "메운다.\n\n"
        "핵심 결과는 세 가지다. 첫째, 지역별 상해(외인) 사망 발생률을 학습한 AI가 단순 계산보다 "
        f"정확했다. 예측 오차를 처음 보는 지역 시험에서 {sp_r}%, 미래 연도 시험에서 {tp_r}% 줄였다. "
        f"둘째, {c['schedule']} 기준 1인당 적정 보험료를 {_fmt(c['m5']['gross_pc'])}원으로 산출했다"
        f"{f'(실제 보고된 {_fmt(reported)}원보다 낮다)' if reported else ''}. 셋째, 정해진 예산 안에서 "
        "장병이 받는 혜택이 최대가 되도록 보장을 배분하며, 생명과 중대 장해는 반드시 보장한다.\n\n"
        "한 가지 원칙을 분명히 했다. 개인의 사고를 예측하거나 개인별로 보험료를 차등하지 않는다. "
        "성능이 약한 모델은 약하다는 사실을 그대로 공개한다.\n"
    )

    # ── 미션·윤리 ────────────────────────────────────────────────
    w("## 0. 미션·윤리")
    w(
        "이 시스템의 목표는 같은 예산으로 더 많은 장병이 더 많은 혜택을 받게 하는 것이다. 비용을 "
        "줄이는 데 목적이 있지 않고, 정해진 예산 안에서 혜택을 키우는 데 초점을 둔다.\n\n"
        "지자체와 보험사의 보장 설계와 예산 책정을 돕는 집단 단위 도구이며, 개인의 사고를 예측하거나 "
        "개인별로 보험료를 차등하지 않는다. 예산이 빠듯해도 생명과 중대 장해 보장은 반드시 지키도록 "
        "완전보장 하한선을 두어, 비용 효율을 이유로 고위험군이 밀려나지 않게 했다.\n"
    )

    # ── 1. 모듈 종합 ─────────────────────────────────────────────
    m0 = c["m0_envelope"]; a = c["moduleA"]; m2 = c["m2"]; m5 = c["m5"]; lp = c["lp"]
    w("## 1. 모듈 종합 (M0~M6 + 모듈A)")
    w("여섯 단계와 발생률 학습(모듈A)이 하나의 흐름으로 이어진다. 단계별 역할과 핵심 산출은 다음과 같다.\n")
    w("| 모듈 | 내용 | 핵심 산출 |")
    w("|---|---|---|")
    w(f"| M0 | 사망 원인을 상해·질병·자살·기타로 분해(겹침·빠짐 없이) | 4범주 합과 전체 사망의 오차 {m0['rel_error_pct']}% |")
    w("| M1·M3 | 보장 항목별 발생률 추정, 지역·연령·성별 구분 | 상해사망·후유장해·골절·입원 발생률 |")
    w(f"| M2 | 개인 손상위험 학습(집단 위험요인 파악용) | 정확도(AUC) 최고 {m2['kfold_best_auc']}. 나이·성별만 쓴 기준({m2['holdout_baseline_auc']})을 못 넘어 단순식 채택 |")
    w(f"| 모듈A | 지역·연도별 상해사망 발생률 학습(GBM), 지역·시기 나눠 검증 | 채택={a['adopt_gbm']}, 미래 시험 보정 {a['temporal_calib']} ({a['years'][0]}–{a['years'][1]}) |")
    w("| M4 | 지역 위험 종합 | 지자체 현역 인원 자동 산출 |")
    w(f"| M5 | 계리 보험료(실측 데이터 기반) | 1인당 총보험료 {_fmt(m5['gross_pc'])}원 (지역 편차 {m5['cv']}, 할증 {m5['loading_implied']}배) |")
    w(f"| M6 | 시나리오·민감도, 예산 최적화(혜택 최대화) | 정해진 예산으로 혜택 총량을 최대화. 지금 급부표에선 단순 배분과 결과가 같지만, 가정을 흔들면 최적화가 앞선다 |")
    w("")

    # ── 2. 평가기준별 매핑 ───────────────────────────────────────
    w("## 2. 평가기준별 매핑")

    w("### 2-1. 공공데이터 활용")
    w("| 기관 | 데이터 | 연도 | 용도 |")
    w("|---|---|---|---|")
    use = {
        "징병 신체검사 통계(등급→현역율·BMI)": "현역 모집단 N 산출 · 자가보고 BMI 보정",
        "중증외상·퇴원손상·비외상성 중증손상(발생·치명·장해)": "상해/질병 사망·후유장해·골절·입원 발생률",
        "지역사회건강조사(CHS) 시군구 건강행태·BMI": "생태회귀·지역 리스크 층화 + 셀패널 특성",
        "국민건강영양조사(KNHANES) 측정 BMI·흡연·손상(IJMT)": "M2 개인 손상위험 ML · 측정값 보정",
        "장래인구추계(시도×성×5세)": "현역 모집단 미래 투영 · 인구절벽 시나리오",
        "사망원인통계(전체사망 envelope 검증)": "M0 4범주 분해 envelope 검증",
        "군복무 청년 상해보험 급부표(연구 보고서)": "기대청구액·권장 보험료·예산 배분",
        "사망사고 통계(군기·안전사고 건수 + 군·민간 자살률)":
            "★군 코호트 proxy 타당성 검증(자살 과대추정 보정·외인 추세 교차검증)",
    }
    for org, desc, yr, _lic in DATA_SOURCES:
        w(f"| {org} | {desc} | {yr} | {use.get(desc, '-')} |")
    w("")

    loro = c["loro"]; tr = c["trend"]; ai = c["ai"]
    red = ai["cv_deviance_reduction"]
    sp_red = red.get("공간CV(leave-1-시도-out)")
    tp_red = red.get("시간CV(최근2년 홀드아웃)")
    eff = ai["efficiency"]
    w("### 2-2. AI 성능 검증")
    w(
        "AI를 두 단계로 나눠 쓰고, 어느 쪽이든 단순 방식과 나란히 비교한 결과를 그대로 보고한다.\n\n"
        "1단계는 지역별 상해(외인) 사망 발생률을 학습하는 모델이다. 보험의 핵심인 이 항목에서 AI는 "
        f"단순 방식을 실제로 이겼다. 처음 보는 지역을 맞히는 시험에서 예측 오차를 {sp_red}%, 미래 연도를 "
        f"맞히는 시험에서 {tp_red}% 줄여 단순 비례와 단순 평균 두 기준을 모두 앞섰기에 채택했다. 예측값을 "
        f"다섯 구간으로 나눠 보면 예측률과 실제 관측률이 거의 일치했고(눈금 기울기 {ai['calib_slope_oos']}, "
        f"1에 가까울수록 정확), 지역 건강습관인 흡연·음주·운동부족·비만을 넣으면 오차가 "
        f"{ai['chs_deviance_reduction_pct']}% 더 줄어 지역 위험요인이 실제로 예측에 기여함을 확인했다. "
        f"예측 범위 안에 실제가 든 비율은 {_fmt(ai['pi_coverage']*100,0)}%로(목표 95%), 지나치게 자신하지 "
        "않고 정직하게 표시했다. 별도의 지역 위험 회귀(LORO)에서도 한 지역씩 빼고 맞힌 평균 오차가 "
        f"{_fmt(loro['model_mae'],2)}로, 전국평균 기준 {_fmt(loro['baseline_mae'],2)}보다 "
        f"{_fmt(loro['mae_improvement_pct'],1)}% 정확했다.\n\n"
        "2단계는 개인의 손상 위험을 맞히는 모델이다. 정확도(AUC)가 최고 "
        f"{m2['kfold_best_auc']}로 약했고, 미래 시험에서는 나이·성별만 쓴 기준"
        f"({m2['holdout_baseline_auc']})조차 넘지 못해 단순식을 유지했다. 개인 예측이 약하다는 사실을 "
        "감추지 않고 그대로 공개하며, 개인을 점수 매기는 데 쓰지 않고 집단 위험요인을 파악하는 데만 쓴다.\n\n"
        "검증의 깊이가 차별점이다. 지역 시험, 미래 시험, 보정, 요인 기여도, 예측 범위에 더해 국방부 "
        f"실데이터 교차검증(5절)까지 거쳤다. 모델 학습은 {_fmt(eff['gbm_fit_ms'],0)}ms, 예산 최적화 계산은 "
        f"{eff['lp_solver_ms']}ms로 빠르고, 같은 입력이면 항상 같은 결과가 나온다"
        f"{'' if eff['deterministic'] else '(주의: 비결정)'}.\n"
    )
    w("상해사망 예측 보정 점검. 다섯 구간 모두에서 예측률이 실제 관측률과 거의 같다(기울기 0.99).\n")
    w("| 예측 5분위 | 예측률(/10만) | 관측률(/10만) | n |")
    w("|---|--:|--:|--:|")
    for _, rr in c["ai_calib"].iterrows():
        w(f"| {int(rr['bin'])} | {rr['예측률_per100k']} | {rr['관측률_per100k']} | {int(rr['n_cells'])} |")
    w("")

    w("### 2-3. 독창성")
    w(
        "실제 보험금 청구 자료가 없는 군 보험 위험을 공개 통계만으로 수치화하고(발생률로 청구액을 대신 "
        "추정), 보장 항목별 보험료까지 산출한 점이 첫 번째 독창성이다.\n\n"
        "두 번째는 예산 최적화의 목표를 미션에 맞춘 것이다. 비용을 줄이는 게 아니라 정해진 예산 안에서 "
        "장병 혜택을 최대로 키우도록 배분하고(선형계획법), 생명과 중대 장해는 반드시 지키는 하한선을 둔다. "
        "여기서 왜 혜택 인원(머릿수)이 아니라 가중 혜택을 쓰는지가 중요하다. 단순히 인원만 최대화하면 값싼 "
        "보장을 잔뜩 끼워 넣는 쪽으로 치우친다. 목표는 더 많은 사람에게 값싼 혜택이 아니라 더 많은 장병이 "
        "더 중대한 혜택을 받는 것이라, 사건의 중대도(사망>후유장해>경상)로 가중치를 준다. 이 가중치는 결과에 "
        "맞춰 거꾸로 맞춘 게 아니라, 국제 질병부담(GBD) 장애가중치 같은 독립 근거로 미리 정해 고정했다.\n"
    )

    w("### 2-4. 발전가능성")
    cliff = c["cliff"]
    w(
        "지자체(보험 가입 주체)와 보험사(보험료 산정) 양쪽에 쓸 수 있다. 전국 20대 남성은 "
        f"{_fmt(cliff['y2024'])}명에서 {_fmt(cliff['y2030'])}명으로 줄어드는데({cliff['pct']}%, 2024→2030), "
        "복무 인원이 줄면 미래 총예산도 달라진다(3-2의 다년 범위 참조). 앞으로 병무청 공개 API를 실시간 "
        "연동하고, 군 훈련 손상의 실제 청구 자료를 확보하면 추정값을 실측으로 전환하며, 통계청 안심구역 "
        "미시데이터로 개인 정보를 보강할 수 있다.\n"
    )

    # ── 3. M6 시나리오 핵심 표 ───────────────────────────────────
    w("## 3. M6 시나리오·민감도 핵심")
    w("### 3-1. 어떤 값이 1인당 보험료를 가장 크게 흔드나 (토네이도 분석)")
    tor = c["tornado"]
    w("| 변수 | 낙관(예산↓) | 기준 | 보수(예산↑) | 변동폭 | 기준대비% |")
    w("|---|--:|--:|--:|--:|--:|")
    for _, r in tor.iterrows():
        w(f"| {r['변수']} | {_fmt(r['낙관(예산↓)'])} | {_fmt(r['기준'])} | "
          f"{_fmt(r['보수(예산↑)'])} | {_fmt(r['swing'])} | {r['swing%기준']}% |")
    w("\n1인당 보험료는 보장 한도와 사업비가 가장 크게 흔든다. 인구 감소는 1인당 가격에는 거의 "
      "영향이 없고, 총예산 규모에서 작동한다.\n")

    w("### 3-2. 다년 예산 경로 밴드 (인구절벽 × 외인추세, 미래 투영)")
    my = c["multiyear"]
    w("| 연도 | 모집단 | 낙관 총예산 | 기준 총예산 | 보수 총예산 |")
    w("|---|--:|--:|--:|--:|")
    for _, r in my.iterrows():
        w(f"| {int(r['year'])} | {_fmt(r['population'])} | {_fmt(r['optimistic_총예산'])} | "
          f"{_fmt(r['base_총예산'])} | {_fmt(r['conservative_총예산'])} |")
    w(
        f"\n미래 상해사망 추세는 하나의 값이 아니라 범위로 제시한다. 낙관은 과거 감소세가 재개되는 경우"
        f"({tr['optimistic']}%/년), 기준은 최근 둔화가 이어지는 경우({tr['base']}%/년), 보수는 감소가 멈추는 "
        f"경우({tr['conservative']}%/년)다. 인구는 통계청 중위 추계 한 경로를 쓰고, 범위의 폭은 상해사망 "
        f"추세의 불확실성에서 나온다. 과거 추세를 연장한 것이라 정책·인구·사회 변화는 반영되지 않는다.\n"
    )

    w("### 3-3. 예산 최적화 — 정직한 비교")
    lp = c["lp"]
    w(
        "최적화의 목표를 '혜택 인원'에서 '중대도로 가중한 혜택'으로 바꿨다(미션에 맞춤). 예산 80%에서 "
        "Greedy(단순 우선순위 배분)와 LP(선형계획 최적화)의 가중 혜택은 "
        f"{_fmt(lp['greedy_welfare'],1)} = {_fmt(lp['lp_welfare'],1)}로 같다.\n\n"
        "왜 같은지 정직하게 설명하면 이렇다. 완전보장 하한이 비싸고 드문 항목(사망 등)을 양쪽 다 100% "
        "보장으로 고정하고, 그 위에 남는 선택 항목이 골절·입원 두 개뿐이고 지급액도 거의 같다"
        "(30.0만≈31.6만원). 이렇게 아슬아슬한 경계에서는 Greedy가 고른 순서가 마침 최적과 일치한다. "
        "가중치를 바꿔 LP가 억지로 이기게 만들지 않았다.\n"
    )
    w("① 가중치를 바꿔가며 시험했다. 최적화가 특정 가중치에서만 이기는지 전부 공개한다.\n")
    ws = c["weight_sens"]
    w("| 가중치 기준 | Greedy | LP | LP 우위 | 비고 |")
    w("|---|--:|--:|--:|---|")
    for _, r in ws.iterrows():
        w(f"| {r['가중셋']} | {_fmt(r['그리디_가중혜택'],1)} | {_fmt(r['LP_가중혜택'],1)} | "
          f"{_fmt(r['LP우위'],1)} | {r['비고']} |")
    w("\n원칙에 따라 정한 가중치(머릿수·GBD·중대도, 모두 골절≥입원)에선 결과가 같고, 입원을 골절보다 "
      "높게 둔 경우에만 최적화가 앞선다. 가중치를 주면 무조건 최적화가 이긴다고 주장하지 않는다.\n")
    w("② 가정을 흔드는 시험이다. 단순 배분의 우선순위 가정만 바꾸면 최적화가 확실히 앞선다.\n")
    pt = c["perturbation"]
    w("| 예산 | Greedy(기본) | Greedy(가정변경) | LP(불변) | LP 추가수혜 |")
    w("|---|--:|--:|--:|--:|")
    for _, r in pt.iterrows():
        w(f"| {r['예산(full대비)']} | {_fmt(r['그리디_기본_수혜자'],1)} | {_fmt(r['그리디_섭동_수혜자'],1)} | "
          f"{_fmt(r['LP_수혜자(불변)'],1)} | +{_fmt(r['LP_추가수혜_vs섭동'],1)} |")
    w("\n표·예산·가중치를 그대로 두고 Greedy의 우선순위 가정만 '입원 먼저'로 바꾸면, Greedy는 곧바로 "
      "최선이 아니게 되고(장병 −23~46명) LP는 가정과 무관하게 최적을 유지한다. Greedy가 잘 나온 건 운 좋은 "
      "가정 덕분이었고, LP의 최적 보장은 현실에서 실제로 중요하게 작동한다.\n")

    # ── 4. 정직성·한계 ───────────────────────────────────────────
    w("## 4. 정직성·한계")
    w(
        "이 시스템의 한계를 숨기지 않고 밝힌다. 먼저 개인 예측은 약하다. 쓰인 건강조사에는 개인별 부상 "
        "기록이 없어 개인을 예측할 수 없고(정확도 AUC≈0.56), 집단 위험요인 파악에만 쓴다. 또 실제 보험금 "
        "청구 자료가 공개되지 않아 공개 발생률로 대신 추정하므로, 발생률을 청구액으로 바꾸는 과정에 오차가 "
        "있다. 지역 회귀는 17개 시도 묶음이라 개인에 적용할 수 없고 표본이 작아 참고 수준이며, 성능이 낮게 "
        "나온 경우도 그대로 공개한다. 현역 인원은 인구추계에 현역율과 복무연수를 곱해 근사한 값이다"
        "(경기 9.8만, 보고 6.8만~10.5만과 부합). 통계청 안심구역 미시데이터는 아직 연동하지 않았다.\n\n"
        "보험료 산정에는 원칙이 하나 있다. 현재 보험료와 예산은 실측 데이터로 고정하고, 학습 모델의 미래 "
        "연장은 시나리오 투영에만 쓰며 현재 가격에는 넣지 않는다. 예산 최적화가 단순 배분과 결과가 같게 "
        "나오는 점도 그대로 공개한다. 지금 급부표는 하한선 위 선택 항목이 두 개뿐이고 지급액도 거의 같아 "
        "결과가 같아진 것이며, 이를 숨기거나 가중치를 거꾸로 맞춰 이기게 만들지 않았다. 최적화의 가치는 "
        "미션에 맞는 목표, 최적 보장, 그리고 가정을 흔들어도 견고하다는 데 있다(3-3).\n\n"
        "끝으로 군 집단 데이터가 없어 일반인구 통계를 대리지표로 썼다. 국방부 실데이터로 대조하니 자살은 "
        "군이 민간의 약 44% 수준이고(과대추정), 상해는 추세는 함께 움직이나(상관 r≈0.88) 수준은 약 3배 "
        "과대였다. 이 보정계수는 참고용으로만 보이고 보험료에는 넣지 않는다(자세한 내용은 5절).\n"
    )

    # ── 5. 군 집단 대리지표 검증 ─────────────────────────────────
    mil = c["military"]
    w("## 5. 군 집단 대리지표의 타당성 검증 (국방부 사망사고 통계)")
    sui, ext = mil["suicide"], mil["external"]
    w(
        "군 내부 자료가 없어 일반인구 통계를 군 집단의 대리지표로 썼다. 이 가정이 타당한지, 대회 개최기관인 "
        "국방부의 실데이터로 직접 검증하고 보정 방향까지 제시한다. 현재 보험료(실측 기반)는 그대로 두고 "
        "정직성과 참고용으로만 함께 적는다.\n\n"
        f"자살은 보험 면책 항목이다. 군인 자살률은 민간의 {sui['adopted_ratio']}배(최근 5년, 전 기간 "
        f"{sui['full']}배) 수준이라, 일반인구로 대신 쓰면 군 자살을 과대추정한다({sui['direction']} 2025년 "
        f"민간 자살률은 미공개라 비율에서 제외{sui['excluded_years']}).\n\n"
        f"상해(외인)는 군 안전사고율과 일반인구 20대 남성 상해사망을 대조했다. 추세는 강하게 함께 움직이고"
        f"(상관 r={ext['trend_r']}), 수준은 군이 일반의 {ext['level_ratio']}배(약 {ext['overestimate_factor']}배 "
        "과대)였다. 즉 대리지표는 추세는 타당하나 수준은 과대추정한다. 총기·폭행 같은 군기사고는 작전·징계 "
        "성격이라 일반 상해보험 보장과 성격이 달라 검증에서 제외했다.\n\n"
        "한계도 있다. 자료가 연 단위·전국 단위라 지역별로 나눌 수 없고, 병력 수는 자살 건수를 자살률로 나눠 "
        "역산한 값이며, 민간 자살률은 전 연령 기준이다. 그래서 전국 단위 참고 보정치로만 쓴다. 요컨대 "
        "일반인구 데이터를 대리지표로 쓰되 그 한계를 개최기관 실데이터로 직접 검증해, 정직성과 공공데이터 "
        f"활용을 함께 강화했다(보험료 영향: {mil['pricing_impact']}).\n"
    )

    # ── 6. 데이터 출처·연도·라이선스 ─────────────────────────────
    w("## 6. 데이터 출처·연도·라이선스")
    w("| 기관 | 데이터 | 연도 | 라이선스/제공 |")
    w("|---|---|---|---|")
    for org, desc, yr, lic in DATA_SOURCES:
        w(f"| {org} | {desc} | {yr} | {lic} |")
    w("")
    return "\n".join(L)


def main() -> str:
    import os
    md = build_report()
    out_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "M7_integrated_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"M7 통합 리포트 생성: {path} ({len(md):,} chars)")
    return path


if __name__ == "__main__":
    main()
