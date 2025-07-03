# live_broker_core.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid # Added import
# Import the mock API client
from trade_creon_api_mock import CreonAPIMock
logger = logging.getLogger(__name__)
# Configure a logger for this module
# logger = logging.getLogger('LiveBroker')
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

class LiveBroker:
    def __init__(self, api_client: CreonAPIMock):
        self.api_client = api_client
        self.cash = 0.0 # Actual cash, will be synced
        self.positions: Dict[str, Dict[str, Any]] = {} # {stock_code: {'size': int, 'avg_price': float}}
        self.transaction_log: List[Dict[str, Any]] = []
        self.pending_orders: Dict[str, Dict[str, Any]] = {} # Track orders sent to broker

        self.commission_rate = 0.0016 # Example commission (0.16%)

        # Link broker's callback methods to the API client
        self.api_client.set_order_reply_callback(self.on_order_reply)
        self.api_client.set_conclusion_callback(self.on_conclusion)

        self._sync_account_status()
        logger.info(f"LiveBroker initialized. Synced cash: {self.cash:,.0f} KRW, Positions: {self.positions}")

    def _sync_account_status(self):
        """
        Fetches current cash and positions from the actual broker (mocked in this case).
        In a real system, this queries Creon API directly.
        """
        self.cash = self.api_client.get_account_balance()
        # For simplicity, we assume positions are initially empty or fetched/reconciled externally.
        # self.positions = self.api_client.get_current_holdings() 
        logger.info("Account status synced with mock broker.")

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int) -> Optional[str]:
        """
        Sends an actual order to the broker via the (mock) Creon API.
        Does NOT update positions/cash immediately; relies on order confirmations.
        """
        logger.info(f"Attempting to send {order_type.upper()} order for {stock_code}, QTY: {quantity}, Price: {price}")
        
        # Simple check for sufficient cash for BUY orders (before sending)
        if order_type.lower() == 'buy':
            required_cash = price * quantity * (1 + self.commission_rate)
            if self.cash < required_cash:
                logger.error(f"Insufficient cash to buy {quantity} of {stock_code} @ {price}. Need {required_cash:,.0f} KRW, have {self.cash:,.0f} KRW.")
                return None
        
        # Simple check for sufficient stock for SELL orders
        if order_type.lower() == 'sell':
            current_position_size = self.positions.get(stock_code, {}).get('size', 0)
            if current_position_size < quantity:
                logger.error(f"Insufficient {stock_code} shares to sell {quantity}. Have {current_position_size}.")
                return None

        # --- FIX: Generate order ID and add to pending_orders BEFORE calling API ---
        order_id = str(uuid.uuid4()) 
        self.pending_orders[order_id] = {
            'stock_code': stock_code,
            'type': order_type,
            'requested_price': price,
            'requested_quantity': quantity,
            'filled_quantity': 0, # How much has been filled so far
            'status': 'PENDING',
            'order_time': datetime.now()
        }
        logger.info(f"Order {order_id} prepared and tracked as PENDING.")
        # --- END FIX ---

        try:
            # Call the API client's send_order, passing the pre-generated order_id
            self.api_client.send_order(stock_code, order_type, quantity, price, order_id) # Pass order_id
            
            logger.info(f"Order {order_id} sent to mock API. Waiting for callbacks...")
            return order_id
        except Exception as e:
            logger.error(f"Error sending order: {e}", exc_info=True)
            # If sending fails, remove from pending_orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            return None

    def on_order_reply(self, data: Dict[str, Any]):
        """
        Callback handler for order replies (e.g., order accepted, rejected).
        This method is called by the Creon API's event system (mocked here).
        """
        order_id = data.get('order_id')
        status = data.get('order_status')
        stock_code = data.get('stock_code')

        if order_id in self.pending_orders:
            self.pending_orders[order_id]['status'] = status
            logger.info(f"Order Reply: Order ID {order_id} for {stock_code} status: {status}")
            if status == '거부': # 'Rejected'
                logger.warning(f"Order {order_id} rejected for {stock_code}. Removing from pending.")
                del self.pending_orders[order_id]
        else:
            logger.warning(f"Order Reply for unknown ID: {order_id} (Status: {status})")

    def on_conclusion(self, data: Dict[str, Any]):
        """
        Callback handler for order execution confirmations (trade fills).
        This method is called by the Creon API's event system (mocked here).
        This is where actual balance updates happen.
        """
        order_id = data.get('order_id')
        stock_code = data.get('stock_code')
        trade_type = 'buy' if data.get('side') == '1' else 'sell'
        filled_quantity = data.get('quantity')
        filled_price = data.get('price')
        trade_time = data.get('timestamp')

        logger.info(f"Order Conclusion: Order ID {order_id} - {trade_type.upper()} {filled_quantity} shares of {stock_code} @ {filled_price} at {trade_time}")

        # Update pending order status and filled quantity
        if order_id in self.pending_orders:
            pending_order = self.pending_orders[order_id]
            pending_order['filled_quantity'] += filled_quantity
            
            # Calculate commission and net amount
            transaction_value = filled_price * filled_quantity
            commission = transaction_value * self.commission_rate
            net_amount = transaction_value + commission if trade_type == 'buy' else transaction_value - commission # Commission paid on buy, deducted from sell
            
            # Update cash and positions
            if trade_type == 'buy':
                self.cash -= net_amount
                # Update position: add shares, recalculate average price
                if stock_code not in self.positions:
                    self.positions[stock_code] = {'size': 0, 'avg_price': 0.0}
                current_size = self.positions[stock_code]['size']
                current_value = current_size * self.positions[stock_code]['avg_price']
                
                new_size = current_size + filled_quantity
                new_value = current_value + transaction_value
                new_avg_price = new_value / new_size if new_size > 0 else 0.0
                
                self.positions[stock_code]['size'] = new_size
                self.positions[stock_code]['avg_price'] = new_avg_price

            elif trade_type == 'sell':
                self.cash += net_amount
                # Update position: subtract shares
                if stock_code in self.positions:
                    self.positions[stock_code]['size'] -= filled_quantity
                    if self.positions[stock_code]['size'] <= 0:
                        del self.positions[stock_code] # Remove if position is zero or negative

            # Log the transaction
            self.transaction_log.append({
                'order_id': order_id,
                'time': trade_time,
                'stock_code': stock_code,
                'type': trade_type,
                'price': filled_price,
                'quantity': filled_quantity,
                'commission': commission,
                'net_cash_change': -net_amount if trade_type == 'buy' else net_amount,
                'cash_after': self.cash
            })
            
            logger.info(f"Balance Update: Cash: {self.cash:,.0f} KRW, {stock_code} holdings: {self.positions.get(stock_code, {}).get('size', 0)} shares.")

            # Check if order is fully filled
            if pending_order['filled_quantity'] >= pending_order['requested_quantity']:
                pending_order['status'] = 'FILLED'
                del self.pending_orders[order_id]
                logger.info(f"Order {order_id} for {stock_code} fully filled.")
            else:
                pending_order['status'] = 'PARTIALLY_FILLED'
                logger.info(f"Order {order_id} for {stock_code} partially filled. Remaining: {pending_order['requested_quantity'] - pending_order['filled_quantity']}")

        else:
            logger.warning(f"Conclusion for unknown or already processed order ID: {order_id}")

    def get_current_cash(self) -> float:
        return self.cash

    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions

    def get_transaction_log(self) -> List[Dict[str, Any]]:
        return self.transaction_log