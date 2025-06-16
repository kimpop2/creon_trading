"""
전략 파라미터 최적화 테스트 스크립트
듀얼 모멘텀과 RSI 전략의 파라미터를 최적화하고 결과를 분석합니다.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from optimizer.strategy_optimizer import StrategyOptimizer
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute
from optimizer.optimizer_broker import OptimizerBroker
from backtest.backtester import Backtester
from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager

def setup_logging():
    """로깅 설정"""
    log_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'strategy_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 로그 포맷 설정
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 콘솔 핸들러 설정 (더 자세한 정보 출력)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', date_format))
    console_handler.setLevel(logging.INFO)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def create_parameter_grids():
    """최적화할 파라미터 그리드 생성 (축소된 범위)"""
    return {
        'daily': {
            'momentum_period': [60, 120],           # 2가지로 축소
            'num_top_stocks': [3, 5],               # 2가지로 축소
            'rebalance_weekday': [0, 1],            # 2가지로 축소
            'safe_asset_code': ['A114260']          # KODEX 레버리지 (A 접두어 추가)
        },
        'minute': {
            'minute_rsi_period': [14],              # 1가지로 축소
            'minute_rsi_oversold': [20, 30],        # 유지
            'minute_rsi_overbought': [70, 80],      # 유지
            'stop_loss_pct': [0.03, 0.05],          # 2가지로 축소
            'portfolio_stop_loss_pct': [0.05, 0.07] # 유지
        }
    }

def run_optimization():
    """전략 파라미터 최적화 실행"""
    logger = setup_logging()
    logger.info("="*50)
    logger.info("전략 파라미터 최적화를 시작합니다.")
    logger.info("="*50)
    
    try:
        # Creon API 클라이언트 초기화
        logger.info("\n1. Creon API 연결 중...")
        api_client = CreonAPIClient()
        if not api_client.connected:
            logger.error("Creon API 연결 실패. 프로그램을 종료합니다.")
            return None
        logger.info("Creon API 연결 성공")
        
        # 테스트 기간 설정 (최근 3개월)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3개월로 축소
        fetch_date = start_date - timedelta(days=30)
        logger.info(f"\n2. 테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 브로커 초기화
        initial_cash = 100000000  # 1억원
        broker = OptimizerBroker(initial_cash=initial_cash)
        
        # 백테스터 생성 (실제 API 클라이언트 사용)
        logger.info("\n3. 백테스터 초기화 중...")
        backtester = Backtester(api_client=api_client, initial_cash=initial_cash)
        backtester.is_test_mode = False  # 실제 API 모드로 설정
        
        # 전략 클래스 정의
        daily_strategies = {'DualMomentumDaily': DualMomentumDaily}
        minute_strategies = {'RSIMinute': RSIMinute}
        # 테스트 종목 리스트 설정
        test_stocks = {
            '반도체': [
                ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT'),
                ('네패스아크', 'IT'), ('와이아이케이', 'IT')
            ],
            # '2차전지': [
            #     ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지'),
            #     ('에코프로비엠', '2차전지'), ('포스코퓨처엠', '2차전지'), ('LG화학', '2차전지'),
            #     ('일진머티리얼즈', '2차전지'), ('엘앤에프', '2차전지')
            # ],
            # '바이오': [
            #     ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오'),
            #     ('유한양행', '바이오'), ('한미약품', '바이오')
            # ],
            # '플랫폼/인터넷': [
            #     ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT'),
            #     ('엔씨소프트', 'IT'), ('넷마블', 'IT')
            # ],
            # '자동차': [
            #     ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차'),
            #     ('만도', '자동차'), ('한온시스템', '자동차')
            # ],
            # '철강/화학': [
            #     ('POSCO홀딩스', '철강'), ('고려아연', '철강'), ('롯데케미칼', '화학'),
            #     ('금호석유', '화학'), ('효성첨단소재', '화학')
            # ],
            # '금융': [
            #     ('KB금융', '금융'), ('신한지주', '금융'), ('하나금융지주', '금융'),
            #     ('우리금융지주', '금융'), ('메리츠금융지주', '금융')
            # ],
            # '통신': [
            #     ('SK텔레콤', '통신'), ('KT', '통신'), ('LG유플러스', '통신'),
            #     ('SK스퀘어', '통신')
            # ],
            # '유통/소비재': [
            #     ('CJ제일제당', '소비재'), ('오리온', '소비재'), ('롯데쇼핑', '유통'),
            #     ('이마트', '유통'), ('BGF리테일', '유통')
            # ],
            # '건설/기계': [
            #     ('현대건설', '건설'), ('대우건설', '건설'), ('GS건설', '건설'),
            #     ('두산에너빌리티', '기계'), ('두산밥캣', '기계')
            # ],
            # '조선/항공': [
            #     ('한국조선해양', '조선'), ('삼성중공업', '조선'), ('대한항공', '항공'),
            #     ('현대미포조선', '조선')
            # ],
            # '에너지': [
            #     ('한국전력', '에너지'), ('한국가스공사', '에너지'), ('두산퓨얼셀', '에너지'),
            #     ('에스디바이오센서', '에너지')
            # ],
            # '반도체장비': [
            #     ('원익IPS', 'IT'), ('피에스케이', 'IT'), ('주성엔지니어링', 'IT'),
            #     ('테스', 'IT'), ('에이피티씨', 'IT')
            # ],
            # '디스플레이': [
            #     ('LG디스플레이', 'IT'), ('덕산네오룩스', 'IT'), ('동운아나텍', 'IT'),
            #     ('매크로젠', 'IT')
            # ],
            '방산': [
                ('한화에어로스페이스', '방산'), ('LIG넥스원', '방산'), ('한화시스템', '방산'),
                ('현대로템', '방산')
            ]
        }
        data_manager = DataManager()
        # 안전자산 코드도 미리 추가
        #safe_asset_code = daily_strategies.strategy_params['safe_asset_code'] # 듀얼 모멘텀 전략의 안전자산 코드 사용
        #logging.info(f"'안전자산' (코드: {safe_asset_code}) 안전자산 일봉 데이터 로딩 중... (기간: {fetch_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
        #daily_df = data_manager.cache_daily_ohlcv(safe_asset_code, fetch_date, start_date, end_date)
        #backtester.add_daily_data(safe_asset_code, daily_df)
        # if daily_df.empty:
        #     logging.warning(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터를 가져올 수 없습니다. 종료합니다.")
        #     exit(1)
        # logging.debug(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
        
        logger.info("\n테스트 종목:")
        for code, name in test_stocks.items():
            logger.info(f"- {code}: {name}")

        # 모든 종목을 하나의 리스트로 변환
        stock_names = []
        for sector, stocks in test_stocks.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)

        # 모든 종목 데이터 로딩
        all_target_stock_names = stock_names
        for name in all_target_stock_names:
            code = api_client.get_stock_code(name)
            if code:
                logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
                daily_df = data_manager.cache_daily_ohlcv(code, fetch_date, end_date)
                
                if daily_df.empty:
                    logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                    continue
                logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
                backtester.add_daily_data(code, daily_df)
            else:
                logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

        if not backtester.data_store['daily']:
            logging.error("백테스트를 위한 유효한 일봉 데이터가 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)


        # 파라미터 그리드 생성
        param_grids = create_parameter_grids()
        
        # 총 조합 수 계산 및 로깅
        total_combinations = (
            len(param_grids['daily']['momentum_period']) *
            len(param_grids['daily']['num_top_stocks']) *
            len(param_grids['daily']['rebalance_weekday']) *
            len(param_grids['minute']['minute_rsi_period']) *
            len(param_grids['minute']['minute_rsi_oversold']) *
            len(param_grids['minute']['minute_rsi_overbought']) *
            len(param_grids['minute']['stop_loss_pct']) *
            len(param_grids['minute']['portfolio_stop_loss_pct'])
        )
        logger.info(f"\n4. 최적화 설정:")
        logger.info(f"- 총 테스트할 파라미터 조합 수: {total_combinations}")
        logger.info("- 예상 소요 시간: 약 7-8분")
        
        # 옵티마이저 초기화
        logger.info("\n5. 최적화 시작...")
        optimizer = StrategyOptimizer(backtester=backtester, initial_cash=initial_cash)
        
        # 진행률 표시를 위한 tqdm 설정
        with tqdm(total=total_combinations, desc="최적화 진행률") as pbar:
            def progress_callback(current, total):
                pbar.update(1)
            
            optimization_result = optimizer.optimize(
                daily_strategies=daily_strategies,
                minute_strategies=minute_strategies,
                param_grids=param_grids,
                start_date=start_date,
                end_date=end_date
            )
        
        if optimization_result is None:
            logger.error("\n최적화 실패")
            return None
            
        # 최적화 결과 분석 및 로깅
        logger.info("\n" + "="*50)
        logger.info("최적화 결과")
        logger.info("="*50)
        
        # 최적 전략 조합 로깅
        best_combination = optimization_result.get('best_combination', {})
        logger.info(f"\n최적 전략 조합:")
        logger.info(f"- 일봉 전략: {best_combination.get('daily', 'N/A')}")
        logger.info(f"- 분봉 전략: {best_combination.get('minute', 'N/A')}")
        
        # 최적 파라미터 로깅
        best_params = optimization_result.get('best_params', {})
        logger.info(f"\n최적 일봉 전략 파라미터:")
        for param, value in best_params.get('daily_params', {}).items():
            logger.info(f"- {param}: {value}")
            
        logger.info(f"\n최적 분봉 전략 파라미터:")
        for param, value in best_params.get('minute_params', {}).items():
            logger.info(f"- {param}: {value}")
            
        # 성과 지표 로깅
        best_metrics = optimization_result.get('best_metrics', {})
        logger.info(f"\n최고 성과 지표: {best_metrics.get('sharpe_ratio', 0):.4f}")
        
        # 상세 성과 지표 로깅
        logger.info("\n상세 성과 지표:")
        logger.info(f"- 총 수익률: {best_metrics.get('total_return', 0):.2%}")
        logger.info(f"- 샤프 비율: {best_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"- 최대 낙폭: {best_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"- 승률: {best_metrics.get('win_rate', 0):.2%}")
        logger.info(f"- 평균 수익: {best_metrics.get('avg_profit', 0):.2%}")
        logger.info(f"- 평균 손실: {best_metrics.get('avg_loss', 0):.2%}")
        
        # 테스트 통계 로깅
        logger.info("\n테스트 통계:")
        logger.info(f"- 총 테스트 수: {optimization_result.get('total_tests', 0)}")
        logger.info(f"- 성공한 테스트: {optimization_result.get('successful_tests', 0)}")
        logger.info(f"- 실패한 테스트: {optimization_result.get('failed_tests', 0)}")
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"\n최적화 중 오류 발생: {str(e)}", exc_info=True)
        return None

if __name__ == '__main__':
    print("\n전략 최적화 프로그램을 시작합니다...")
    result = run_optimization()
    if result is not None:
        print("\n" + "="*50)
        print("최적화가 성공적으로 완료되었습니다.")
        print("상세 결과는 로그 파일을 확인해주세요.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("최적화가 실패했습니다.")
        print("로그 파일을 확인하여 오류 내용을 확인해주세요.")
        print("="*50) 