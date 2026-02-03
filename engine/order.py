from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(Enum):
    BID = "bid"
    ASK = "ask"

class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"

class OrderStatus(Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = field(default_factory=time.time_ns)
    trader_id: str = ""
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
