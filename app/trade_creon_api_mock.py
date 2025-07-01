# creon_api_mock.py

import logging
import uuid # Added import
import time
from datetime import datetime
from typing import Dict, Any, Callable, Optional # Added Optional
logger = logging.getLogger(__name__)
# Configure a logger for this module
# logger = logging.getLogger('CreonAPIMock')
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

class CreonAPIMock:
    """
    A simplified mock of CreonAPIClient for demonstration purposes.
    It simulates sending orders and firing back order reply/conclusion events.
    """
    def __init__(self):
        self.order_reply_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.conclusion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        logger.info("Mock Creon API Client initialized.")

    def set_order_reply_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Sets the callback for order replies."""
        self.order_reply_callback = callback

    def set_conclusion_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Sets the callback for order conclusions (executions)."""
        self.conclusion_callback = callback

    # Modified: Added 'order_id' as an optional parameter
    def send_order(self, stock_code: str, order_type: str, quantity: int, price: float, order_id: Optional[str] = None) -> str:
        """
        Simulates sending an order to the broker.
        Immediately fires an 'order reply' and then a 'conclusion' after a short delay.
        """
        if order_id is None: # Use provided ID or generate a new one
            order_id = str(uuid.uuid4()) # Generate a unique order ID
            
        logger.info(f"Mocking order sent: ID={order_id}, {order_type} {quantity} {stock_code} @ {price}")

        # Simulate immediate order acceptance (td0314 reply)
        if self.order_reply_callback:
            reply_data = {
                'order_id': order_id,
                'order_status': '접수', # 'Accepted'
                'stock_code': stock_code
            }
            self.order_reply_callback(reply_data)
        
        # Simulate execution after a small delay (conclusion event)
        # In a real system, this would be an actual callback from Creon's CpEvent
        # This part is illustrative of the asynchronous nature.
        def _simulate_execution():
            time.sleep(0.5) # Simulate network latency/execution time
            if self.conclusion_callback:
                conclusion_data = {
                    'order_id': order_id,
                    'stock_code': stock_code,
                    'side': '1' if order_type.lower() == 'buy' else '2', # '1' for buy, '2' for sell
                    'quantity': quantity, # Assume full fill for simplicity
                    'price': price, # Assume filled at requested price for simplicity
                    'timestamp': datetime.now()
                }
                self.conclusion_callback(conclusion_data)
                logger.info(f"Mock conclusion fired for Order ID: {order_id}")

        # In a real async system, you might use threading.Thread or asyncio.create_task here
        # For this simple example, we'll just call it directly to keep it synchronous in the demo.
        # But conceptually, this would be triggered externally by the real API's event loop.
        _simulate_execution() 
        
        return order_id

    def get_account_balance(self) -> float:
        """Simulates fetching current cash balance."""
        return 10_000_000 # Mock initial balance

    def get_current_holdings(self) -> Dict[str, Dict[str, Any]]:
        """Simulates fetching current stock holdings."""
        return {} # Mock initially empty holdings

    def connect(self):
        logger.info("Mock Creon API connected.")
        
    def cleanup(self):
        logger.info("Mock Creon API cleaned up.")