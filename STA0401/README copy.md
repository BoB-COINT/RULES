# Honeypot Detection Pipeline

ERC-20 허니팟/스캠 토큰을 온체인 이벤트 기반으로 분석하고, 전처리된 **features.csv** 위에 룰(**honeypot.yaml**)을 적용해 탐지 결과(**detections.json**)를 생성하는 파이프라인입니다.

---

## 빠른 시작 (Quick Start)

### 0) 요구 사항
- 기본 패키지: `web3`, `pandas`, `python-dotenv` 등  
- 환경변수:
  - `ALCHEMY_KEY`
  - `ETHERSCAN_KEY`

---

## 파이프라인

### 1) 데이터셋 수집

- **입력**: `{token_list}.json`  
  - 분석 대상 토큰 주소 목록을 담은 JSON

### 2) 원천데이터 생성

- **관련 코드**:  
  `/module/__init__.py`, `TokenContEventParser.py`, `TokenPairEventParser.py`  
  `/main_cont.py`, `main_pair.py`

- **실행 예시 (PowerShell)**:
  ```bash
  # 토큰 컨트랙트 이벤트 수집
  py .\main_cont.py ".\datasets\{token_list}.json"

  # DEX 페어/풀 이벤트 수집
  py .\main_pair.py ".\datasets\{token_list}.json"
  ```

- **결과물**:
  - `token_evt.csv`
  - `pair_evt.csv`
  - `token_information.csv`

### 3) 전처리 (Feature 생성)

```bash
python generate_features_STA0401.py
```

- **입력**: `token_evt.csv`, `pair_evt.csv`  
- **출력**: `features.csv`

### 4) 룰 적용 (허니팟 탐지)

```bash
python run_rules_honeypot.py `
  --features ./features.csv `
  --rule ./honeypot.yaml `
  --output-json ./output/detections.json
```

- **입력**: `features.csv`, `honeypot.yaml`  
- **출력**: `./output/detections.json`

---

## 코드 개요 (Code)

### 원천데이터

#### `main_cont.py` (토큰 컨트랙트 이벤트)

```bash
main_cont.py : 토큰 컨트랙트 이벤트 
├─ 1. 환경변수 확인 
├─ 2. RPC 연결 
├─ 3. **입력값** : JSON(데이터셋) 로드 
├─ 4. csv 준비 
├─ 5. 토큰 순회 - TokenContEventParser.py
│    ├─ parser() -> 토큰 정보 수집 
│    │            -> 종료 블록 계산(생성 후 14일) 
│    │            -> 이벤트 로그(Transfer, Approval) 수집
└── **결과물** : token_evt.csv 
```

- `token_evt.csv` 스키마
| 컬럼 |
|---|
| { token_addr_idx, timestamp, block_number, tx_hash, tx_from, tx_to, evt_idx, evt_type, evt_log } |

#### `main_pair.py` (DEX Pair/Pool 이벤트)

```bash
main_pair.py : DEX Pair 이벤트 
├─ 1. 환경변수 확인 
├─ 2. RPC 연결 
├─ 3. **입력값** : JSON(데이터셋) 로드 
├─ 4. 토큰 순회 - TokenPairEventParser.py
│    ├─ parser() -> 토큰의 모든 Pair 조회 
│    │            -> 각 Pair의 생성 정보 수집 
│    │            -> 토큰마다 Pair 이벤트(Swap, Mint, Burn, Sync) 수집
└── **결과물** : pair_evt.csv, token_information.csv  
```

- `pair_evt.csv` 스키마
| 컬럼 |
|---|
| { token_addr_idx, timestamp, block_number, tx_hash, tx_from, tx_to, evt_idx, evt_type, evt_log, token0, token1, reserve0, reserve1, lp_total_supply, evt_removed } |

- `token_information.csv` 스키마
| 컬럼 |
|---|
| { token_addr_idx, token_addr, pair_addr, token_create_ts, lp_create_ts, pair_idx } |

---

## 전처리 (Feature 생성)

#### `generate_features_0401_v8.py`

```bash
generate_features_0401_v8.py
├─ 1. 환경 설정
│   ├─ BASE_DIR 설정
│   ├─ **입력값** : token_evt.csv, pair_evt.csv
│
├─ 2. 데이터 로드 및 검증 
│   ├─ token_evt : Transfer/Approval 이벤트 파싱
│   ├─ pair_evt : Swap/Mint/Burn/Sync 이벤트 파싱
│
├─ 3. 토큰별 Feature 생성 루프 
│   ├─ 3-1. S_owner 식별 : 초기 민팅 수령자, LP 민터/버너 추출 
│   ├─ 3-2. 윈도우별 Feature : 5초 단위, buy/sell 탐지, 오너/일반인 구분 등 
│   ├─ 3-3. 토큰별 집계 
│
└── **결과물** : features.csv 
```

- **S_owner**: “초기 민팅 수령자 + LP 민팅/소각 참여자”로 정의된 **특권/오너 후보 주소 집합**

- `features.csv` 스키마
| 컬럼 |
|---|
| { token_addr_idx, consecutive_sell_fail_windows, total_buy_cnt, total_sell_cnt, total_owner_sell_cnt, total_non_owner_sell_cnt, owner_sell_ratio, total_approval_cnt, imbalance_rate, approval_to_sell_ratio, failed_sell_proxy_frac, max_sell_share, privileged_event_flag, router_only_sell_proxy, total_windows, windows_with_activity, total_burn_events, total_mint_events, avg_burn_frac, avg_reserve_drop, s_owner_count, total_sell_vol, total_owner_sell_vol, owner_sell_vol_ratio, router_approval_rate } |

---

## 룰 (STA0401: Honeypot)

#### `run_rules_honeypot.py` (탐지기)

```bash
run_rules_honeypot.py : 허니팟 탐지 룰 적용기
├─ 1. features.csv 로드
├─ 2. rule.yml 로드
├─ 3. Helper 함수로 컬럼/파라미터 보정
├─ 4. 탐지 규칙 평가 (evaluate_rule_sta0401)
│    ├─ 조건 계산
│    ├─ 점수화(score 계산)
│    ├─ 판정 및 심각도 분류
│    └─ Validation 필터링
├─ 5. 탐지 결과 저장 (detections.json)
└─ 6. 전체 실행 (CLI 인자 기반)
```

- 룰 파일: `honeypot.yaml`  
- 결과 파일: `output/detections.json`

- `detections.json` 필드(요약)
| 컬럼 |
|---|
| { token_addr_idx, consecutive_sell_fail_windows, total_buy_cnt, total_sell_cnt, total_owner_sell_cnt, total_non_owner_sell_cnt, owner_sell_ratio, total_approval_cnt, imbalance_rate, approval_to_sell_ratio, failed_sell_proxy_frac, max_sell_share, privileged_event_flag, router_only_sell_proxy, total_windows, windows_with_activity, total_burn_events, total_mint_events, avg_burn_frac, avg_reserve_drop, s_owner_count, total_sell_vol, total_owner_sell_vol, owner_sell_vol_ratio, router_approval_rate, bonus_hits, score, detected, severity } |

> **주의**: 실제 임계값은 `honeypot.yaml`에 정의된 파라미터에 따릅니다.

---

## 데이터 세트 (Test Sets)

- `test0_normal` : 정상 케이스 **288개**
- `test1_case_11pcs` : 케이스 스터디 중 **허니팟 태깅 11개**
- `test2_honeypot_8pcs` : **8개** (trapdoor 태깅 4개 + Etherscan honeypot 태그 4개)
- `test3_defi_honeypot` : DeFi DB에서 수집한 **24개** 케이스

---

## 디렉토리 예시

```
.
├─ module/
│  ├─ __init__.py
│  ├─ TokenContEventParser.py
│  └─ TokenPairEventParser.py
├─ main_cont.py
├─ main_pair.py
├─ generate_features_0401.py
├─ run_rules_honeypot.py
├─ honeypot.yaml
├─ datasets/
│  ├─ test0_normal/
│  ├─ test1_case_11pcs/
│  ├─ test2_honeypot_8pcs/
│  └─ test3_defi_honeypot/
├─ outputs/
│  └─ detections.json
└─ (생성물)
   ├─ token_evt.csv
   ├─ pair_evt.csv
   ├─ token_information.csv
   └─ features.csv
```
