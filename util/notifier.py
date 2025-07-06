# util/notifier.py

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self, telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        if self.telegram_token and self.telegram_chat_id:
            logger.info("텔레그램 알림 기능이 활성화되었습니다.")
        else:
            logger.warning("텔레그램 토큰 또는 채팅 ID가 설정되지 않아 텔레그램 알림 기능이 비활성화됩니다.")

    def send_message(self, message: str) -> bool:
        """
        지정된 채널로 메시지를 전송합니다.
        현재는 텔레그램만 지원합니다.
        """
        if self.telegram_token and self.telegram_chat_id:
            return self._send_telegram_message(message)
        else:
            logger.info(f"알림 (비활성화): {message}") # 텔레그램 설정이 없으면 로그로만 출력
            return False

    def _send_telegram_message(self, message: str) -> bool:
        """
        텔레그램 봇을 통해 메시지를 전송합니다.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("텔레그램 토큰 또는 채팅 ID가 없어 메시지를 전송할 수 없습니다.")
            return False

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML" # HTML 형식 지원
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status() # HTTP 에러 발생 시 예외 발생
            logger.debug(f"텔레그램 메시지 전송 성공: {message[:50]}...")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"텔레그램 메시지 전송 실패: {e}")
            return False