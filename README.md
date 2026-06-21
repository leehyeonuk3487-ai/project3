# 군복무 청년 상해보험 — 리스크 층화·계리 추정

지자체가 운영하는 **군복무 청년 상해보험**의 보장 설계·예산 책정을 돕기 위한
데이터 파이프라인이다. 공개 통계를 결합해 **보장항목별 발생률을 추정**하고,
지역·연령·성별·행태에 따라 **집단 리스크를 층화**한다.

> ⚠️ **포지셔닝(정직성 우선).** 이 프로젝트는 *개인의 사고를 예측*하지 않는다.
> 사용하는 조사 데이터(CHS/KNHANES)는 횡단면 자료라 개인 단위 부상 결과 라벨이
> 없으며, 따라서 개인 인과 추정도 불가능하다. 대신 공개 발생률 통계(KDCA/KOSIS)와
> 위험요인 분포(CHS)를 결합한 **집단 단위 계리 추정 + 리스크 층화**를 제공한다.
> 모든 산출물에는 이 한계가 함께 표기된다.

---

## 데이터 (모두 실데이터, 저장소에 포함)

| 경로 | 내용 | 역할 |
|---|---|---|
| `data/processed/chs_filtered_2021_2025.csv` | 지역사회건강조사 개인 232,810행 (시도·시군구·행태·BMI·가중치) | 위험요인 노출 분포 |
| `data/processed/knhanes_filtered_2021_2024.csv` | 국민건강영양조사 개인 6,135행 (BMI·혈압 측정값) | 위험요인 보정·검증 |
| `data/raw/kdca_severe_trauma/` | 중증외상 발생률·치명률·장해율 (성/연령/시도) | 사건 발생률 |
| `data/raw/kdca_non_trauma_severe/` | 비외상 중증질환 발생률 | 질병 보장 |
| `data/raw/kdca_discharge_injury/` | 퇴원손상 퇴원율·손상양상(골절 등) | 골절·입원 발생률 |
| `data/raw/kosis_population/` | 시도×성×5세 인구추계 2020–35 | 분모(population at risk) |
| `data/raw/kosis_death_cause/` | 시도×성×연령 사망원인 (cp949 인코딩) | 사망 보장 검증 |
| `data/raw/mma_judgment_status/` | 병무청 판정현황(BMI·등급·신체) + 연감 PDF | 피보험 모집단(현역 청년) |
| `docs/research/local_gov_insurance_benefits.md` | 10개 지자체 보장항목·금액 실태 | 예산 최적화의 급부 스케줄 |

데이터 인코딩이 섞여 있다(CHS/KNHANES=utf-8-sig, 대부분 KDCA=utf-8,
KOSIS 사망원인=cp949). `src/data/encoding.py`가 자동 판별한다.

---

## 방법론

KDCA 발생률표는 성·연령·시도가 **각각 주변분포**로만 제공된다(joint cell 없음).
계리에서 표준적으로 쓰는 **비례 보정(주변 독립 가정)** 으로 결합한다:

```
rate(sex, age, sido) = national_rate × R_sex × R_age × R_sido
R_x = rate(level x) / national_rate
```

퇴원손상표는 성×연령이 **진짜 교차표**라 그 부분은 보정 없이 직접 쓰고,
지역 축만 중증외상 시도 상대위험을 proxy로 적용한다. 골절은
`전체 퇴원율 × 골절분율`로 근사한다.

비례 보정은 근사이므로(상호작용 무시), slice 3에서 KDCA 관측 주변분포 대비
캘리브레이션으로 검증한다.

---

## 구조

```
src/
├── config.py            # 경로, 시도 코드↔명 crosswalk, 보장항목·연령대 상수
├── data/
│   ├── encoding.py      # 인코딩 자동 판별(utf-8-sig/utf-8/cp949)
│   ├── reshape.py       # KDCA/KOSIS 2중헤더 와이드 피벗 → long 변환
│   └── loaders.py       # CHS/KNHANES 개인 + KDCA 발생률 로더(BMI 파생 포함)
└── models/
    └── rates.py         # 보장항목별 성×연령×시도 발생률 surface
scripts/
└── run_slice1.py        # end-to-end 실행 → outputs/conscript_rate_table.csv
```

## 실행

```bash
pip install -r requirements.txt
python -m scripts.run_slice1
```

현역 청년(20대 남성)의 보장항목별 시도 발생률 표와 지역 격차 요약이 출력되고
`outputs/conscript_rate_table.csv` 로 저장된다. 지역 격차는 최대 3.2배로,
지역 보정의 실효성을 뒷받침한다.

---

## 로드맵

- **slice 1 (완료)** — 데이터 로딩/정제 + 보장항목별 기준 발생률 surface.
- **slice 2** — CHS 위험요인 ↔ KDCA 발생률 **생태학적 포아송 회귀**로 상대위험
  추정 → 개인 리스크 **층화(저/중/고)** + 구간 IRR. 치명률·장해율 분율을 결합해
  상해사망·후유장해 분해. 지자체 급부표 구조화 + 예산 최적화.
- **slice 3** — LORO(leave-one-region-out) 캘리브레이션 검증 리포트 +
  FastAPI + 단일 HTML 대시보드.

## 한계

- 개인 부상 결과 라벨이 없어 **개인 예측 불가** — 집단 층화·계리 추정에 한정.
- 발생률 결합은 주변 독립 가정의 근사(상호작용 미반영).
- 상해사망/후유장해 분해, 사망원인 결합은 slice 2 예정(현재는 기준 발생률만).
- 병무청 신체검사 원시 질환 데이터(데이터안심구역)는 SDC 미제공으로 미사용.
