# setup/setup_daily_universe.py 파일 전체를 아래 코드로 교체하세요.

import logging
from datetime import date, timedelta
from time import time
import sys
import os

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 필요한 Manager 클래스 임포트
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.setup_manager import SetupManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DailyUniverseSetup')

def run_universe_selection_process(target_num_stocks: int = 100):
    """
    [수정됨] 최종 유니버스 선정 시 종목 중복을 제거하는 로직이 추가된 전체 프로세스.
    """
    logger.info("========== 🚀 당일 최종 유니버스 선정 프로세스 시작 (Target: daily_universe) ==========")
    start_time = time()

    api_client = None
    db_manager = None

    try:
        # --- 1. 필수 모듈 초기화 ---
        logger.info("[1/4] API 및 DB Manager 초기화...")
        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결에 실패했습니다.")
        
        db_manager = DBManager()
        setup_manager = SetupManager(api_client, db_manager)
        logger.info("초기화 완료.")

        # --- 2. 테마 모멘텀 점수 계산 ---
        logger.info("[2/4] 테마 모멘텀 점수 계산 시작...")
        if not setup_manager.calculate_theme_momentum_scores(data_period_days=40):
            logger.error("테마 모멘텀 점수 계산에 실패하여 프로세스를 중단합니다.")
            return
        logger.info("테마 모멘텀 점수 계산 및 DB 업데이트 완료.")
        
        # --- 3. 유니버스 후보군 생성 ---
        logger.info("[3/4] 유니버스 후보군 생성 시작...")
        candidates = setup_manager.generate_universe_candidates(limit_themes=10, limit_stocks_per_theme=20)
        
        if not candidates:
            logger.warning("생성된 유니버스 후보군이 없습니다. 프로세스를 종료합니다.")
            return
        logger.info(f"{len(candidates)}개 테마의 후보군 생성 완료.")

        # --- 4. 최종 유니버스 선정 및 저장 ---
        logger.info(f"[4/4] 최종 유니버스 선정 (상위 {target_num_stocks}개) 및 저장 시작...")
        
        # [핵심 수정] 종목 코드 기준 중복 제거 로직
        unique_stocks = {}
        for theme_data in candidates:
            for stock_info in theme_data['recommended_stocks']:
                stock_code = stock_info.get('stock_code')
                if not stock_code:
                    continue

                # 각 점수를 안전하게 float으로 변환 (None일 경우 0.0)
                stock_score = float(stock_info.get('stock_score', 0.0) or 0.0)
                price_trend_score = float(stock_info.get('price_trend_score', 0.0) or 0.0)
                trading_volume_score = float(stock_info.get('trading_volume_score', 0.0) or 0.0)
                volatility_score = float(stock_info.get('volatility_score', 0.0) or 0.0)
                theme_mention_score = float(stock_info.get('theme_mention_score', 0.0) or 0.0)
                
                # 최종 저장될 종합 점수 계산
                total_score = stock_score + price_trend_score + trading_volume_score + volatility_score + theme_mention_score
                
                # stock_info 딕셔너리에 계산된 점수들 업데이트
                stock_info.update({
                    'theme': theme_data.get('theme_class'),
                    'theme_id': theme_data.get('theme_id'),
                    'stock_score': total_score,
                    'price_trend_score': price_trend_score,
                    'trading_volume_score': trading_volume_score,
                    'volatility_score': volatility_score,
                    'theme_mention_score': theme_mention_score
                })

                # 종목이 이미 unique_stocks에 있다면, 새로 계산된 점수가 더 높을 경우에만 교체
                if stock_code in unique_stocks:
                    if total_score > unique_stocks[stock_code].get('stock_score', 0.0):
                        unique_stocks[stock_code] = stock_info
                else:
                    unique_stocks[stock_code] = stock_info
        
        # 중복이 제거된 후보군 리스트 생성
        all_candidate_stocks = list(unique_stocks.values())

        # 종합 점수(stock_score) 기준으로 내림차순 정렬
        all_candidate_stocks.sort(key=lambda x: x['stock_score'], reverse=True)
        final_universe = all_candidate_stocks[:target_num_stocks]

        if not final_universe:
            logger.warning("최종 선정된 유니버스 종목이 없습니다.")
            return

        # DB 저장을 위해 최종 데이터 포맷팅
        today = date.today()
        data_to_save = [
            {
                'date': today,
                'stock_code': d.get('stock_code'),
                'stock_name': d.get('stock_name'),
                'theme': d.get('theme'),
                'stock_score': d.get('stock_score', 0.0),
                'price_trend_score': d.get('price_trend_score', 0.0),
                'trading_volume_score': d.get('trading_volume_score', 0.0),
                'volatility_score': d.get('volatility_score', 0.0),
                'theme_mention_score': d.get('theme_mention_score', 0.0),
                'theme_id': d.get('theme_id')
            } for d in final_universe
        ]
        
        # DB에 최종 저장
        if db_manager.save_daily_universe(data_to_save):
            logger.info("최종 유니버스 선정 및 저장이 성공적으로 완료되었습니다.")
        else:
            logger.error("최종 유니버스 저장에 실패했습니다.")

    except Exception as e:
        logger.critical(f"유니버스 선정 프로세스 중 치명적인 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()
        
        end_time = time()
        logger.info(f"========== ✅ 당일 최종 유니버스 선정 프로세스 종료 (총 소요 시간: {end_time - start_time:.2f}초) ==========")

if __name__ == '__main__':
    run_universe_selection_process(target_num_stocks=100)