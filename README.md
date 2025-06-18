# Creon Trading - 백테스팅 및 최적화 시스템

2025-06-10 시작

## 📁 프로젝트 구조

```
creon_trading/
├── api/                    # Creon API 연동
├── backtest/              # 백테스터 엔진
├── strategies/            # 전략 라이브러리
├── optimizer/             # 최적화 엔진
│   ├── results/          # 최적화 결과 파일들 (.json, .csv)
│   ├── grid_search_optimizer.py
│   ├── bayesian_optimizer.py
│   └── progressive_refinement_optimizer.py
├── logs/                  # 로그 파일들
├── manager/               # 데이터 및 DB 관리
├── selector/              # 종목 선택기
├── util/                  # 유틸리티
└── run_*.py              # 실행 스크립트들
```

## 📊 파일 관리

### 최적화 결과 파일
- **위치**: `optimizer/results/`
- **파일 형식**: `.json`, `.csv`
- **내용**: 최적화된 파라미터, 성능 지표, 백테스트 결과

### 로그 파일
- **위치**: `logs/`
- **파일 형식**: `.log`
- **내용**: 실행 로그, 오류 메시지, 진행상황

## 🚀 주요 기능

- **다양한 전략**: SMA, RSI, 볼린저밴드, 섹터로테이션 등
- **최적화 방법**: 그리드서치, 베이지안, 하이브리드 최적화
- **실시간 데이터**: Creon API 연동
- **성능 분석**: 샤프지수, MDD, 승률 등 7개 지표

## 📈 최적화 성과 (2024-12-01 ~ 2025-04-01)

- **수익률**: 14.78%
- **샤프지수**: 1.62
- **MDD**: -8.15%
- **승률**: 43.0%
