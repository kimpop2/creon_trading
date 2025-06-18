# 파일명: selector/stock_select.py
# 설명: 종목 선정 로직 구현 (BaseSelector 상속)
# 작성일: 2025-06-17

import datetime
import logging
from typing import List, Dict

from selector.selector import BaseSelector
from manager.data_manager import DataManager
from api.creon_api import CreonAPIClient

logger = logging.getLogger(__name__)

class StockSelector(BaseSelector):
    """
    주어진 설정과 현재 날짜에 따라 백테스트 대상 종목을 선정하는 클래스입니다.
    뉴스매매 또는 테마주매매에 맞는 종목 선정 클래스가 추가될 수 있으므로 BaseSelector를 상속받습니다.
    """
    def __init__(self, data_manager: DataManager, api_client: CreonAPIClient, sector_stocks_config: Dict):
        super().__init__()
        self.data_manager = data_manager
        self.api_client = api_client
        self.sector_stocks_config = sector_stocks_config
        logger.info("StockSelector 초기화 완료.")

    def select_stocks(self, current_date: datetime.date, selection_params: Dict = None) -> List[str]:
        """
        sector_stocks_config에 정의된 종목 이름들을 기반으로 실제 종목 코드를 조회하고 반환합니다.
        향후 selection_params를 활용하여 보다 복잡한 선정 로직을 추가할 수 있습니다.

        Args:
            current_date (datetime.date): 현재 백테스트 날짜 (일봉 데이터 기준).
            selection_params (Dict): 종목 선정에 필요한 추가 파라미터 (현재는 사용하지 않음).

        Returns:
            List[str]: 선정된 종목 코드 리스트.
        """
        logging.info(f"{current_date.isoformat()} - 종목 선정 로직 실행 중...")
        selected_stock_codes = []
        all_target_stock_names = []

        # run_backtest.py의 sector_stocks 구조를 활용하여 모든 종목 이름 수집
        for category, stocks in self.sector_stocks_config.items():
            for stock_name_tuple in stocks:
                # stock_name_tuple은 (종목명, 섹터) 형태일 수 있음
                stock_name = stock_name_tuple[0] if isinstance(stock_name_tuple, tuple) else stock_name_tuple
                all_target_stock_names.append(stock_name)

        # 중복 제거
        all_target_stock_names = list(set(all_target_stock_names))
        
        for name in all_target_stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                selected_stock_codes.append(code)
            else:
                logger.warning(f"종목명 '{name}'에 대한 종목 코드를 찾을 수 없습니다.")
        
        logger.info(f"{current_date.isoformat()} - 총 {len(selected_stock_codes)}개의 종목 선정 완료.")
        return selected_stock_codes