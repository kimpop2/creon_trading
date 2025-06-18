# 파일명: selector/selector.py
# 설명: 종목 선정 전략의 추상 베이스 클래스
# 작성일: 2025-06-17

from abc import ABC, abstractmethod
import datetime
from typing import List, Dict

class BaseSelector(ABC):
    """
    종목 선정 전략의 추상 베이스 클래스입니다.
    모든 구체적인 종목 선정 전략은 이 클래스를 상속받아야 합니다.
    """
    def __init__(self):
        pass

    @abstractmethod
    def select_stocks(self, current_date: datetime.date, selection_params: Dict) -> List[str]:
        """
        주어진 날짜와 선정 파라미터를 기반으로 백테스트 대상 종목 코드 리스트를 선정합니다.
        모든 하위 클래스는 이 메서드를 구현해야 합니다.

        Args:
            current_date (datetime.date): 현재 백테스트 날짜 (일봉 데이터 기준).
            selection_params (Dict): 종목 선정에 필요한 파라미터.

        Returns:
            List[str]: 선정된 종목 코드 리스트.
        """
        pass