import os
import asyncio
import ccxt.async_support as ccxt

from typing import Optional, List, Any
from pydantic import BaseModel
from decimal import Decimal, getcontext, ROUND_DOWN
import pandas as pd
import time
import math

class Info(BaseModel):
    success: bool
    message: str

class UsdcBalance(BaseModel):
    total: float
    free: float
    used: float

class OpenPosition(BaseModel):
    id: Optional[str]
    symbol: str
    timestamp: Optional[int]
    datetime: Optional[str]
    isolated: bool
    hedged: Optional[bool]
    side: Optional[str]
    contracts: float
    contractSize: float
    entryPrice: float
    markPrice: Optional[float]
    notional: float
    leverage: float
    collateral: Optional[float]
    initialMargin: float
    maintenanceMargin: Optional[float]
    initialMarginPercentage: Optional[float]
    maintenanceMarginPercentage: Optional[float]
    unrealizedPnl: float
    liquidationPrice: Optional[float]
    marginMode: Optional[str]
    percentage: float

class Position(BaseModel):
    pair: str
    side: str
    size: float
    usd_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    # liquidation_price: Optional[float]
    margin_mode: str
    leverage: int
    hedge_mode: bool
    open_timestamp: int = 0
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

class OpenOrder(BaseModel):
    id: Optional[str] = None
    clientOrderId: Optional[str] = None
    timestamp: Optional[int] = None
    datetime: Optional[str] = None
    lastTradeTimestamp: Optional[int] = None
    lastUpdateTimestamp: Optional[int] = None
    symbol: Optional[str] = None
    type: Optional[str] = None
    timeInForce: Optional[str] = None
    postOnly: Optional[bool] = None
    reduceOnly: Optional[bool] = None
    side: Optional[str] = None
    price: Optional[float] = None
    triggerPrice: Optional[float] = None
    amount: Optional[float] = None
    cost: Optional[float] = None
    average: Optional[float] = None
    filled: Optional[float] = None
    remaining: Optional[float] = None
    status: Optional[str] = None
    fee: Optional[Any] = None
    trades: List[Any] = []
    fees: List[Any] = []
    stopPrice: Optional[float] = None
    takeProfitPrice: Optional[float] = None
    stopLossPrice: Optional[float] = None

    class Config:
         extra = "allow"

class Order(BaseModel):
    id: str
    pair: str
    type: str
    side: str
    price: float
    size: float
    reduce: bool
    filled: float
    remaining: float
    timestamp: int

class Market(BaseModel):
    internal_pair: str
    base: str
    quote: str
    price_precision: float
    contract_precision: float
    contract_size: float = 1.0
    min_contracts: float = 1.0
    max_contracts: float = float('inf')
    min_cost: float = 0.0
    max_cost: float = float('inf')
    coin_index: int = 0
    market_price: float = 0.0

def number_to_str(n: float) -> str:
    """
    Convert float to string, removing trailing zeros.
    """
    s = format(n, 'f').rstrip('0')
    if s.endswith('.'):
        s = s[:-1]
    return s

def get_price_precision(price: float) -> float:
    """
    Simple helper to guess a rounding step for the given price.
    Example: price=12345 => precision ~ 1 => will round to nearest 1.
    """
    if price <= 0:
        return 1
    log_price = math.log10(price)
    order = math.floor(log_price)
    precision = 10 ** (order - 4)
    return precision

class PerpHyperliquid:
    def __init__(self, public_api=None, secret_api=None):
        self.public_address = public_api
        self.private_key = secret_api

        hyperliquid_auth_object = {
            "walletAddress": self.public_address,
            "privateKey": self.private_key,
        }

        # Increase Decimal precision to handle high-priced assets like BTC
        from decimal import getcontext
        getcontext().prec = 28
        
        if self.private_key is None:
            self._auth = False
            self._dex = ccxt.hyperliquid()
        else:
            self._auth = True
            self._dex = ccxt.hyperliquid(hyperliquid_auth_object)

        self.market: dict[str, Market] = {}

    # -------------------------------------------- #
    #  BALANCE                                     #
    # -------------------------------------------- #
    async def get_balance(self) -> UsdcBalance:
        params = {"vaultAddress": self.public_address}
        data = await self._dex.fetch_balance(params)
        total = float(data["USDC"]["total"])
        used = float(data["USDC"]["used"])
        free = total - used
        return UsdcBalance(total=total, free=free, used=used)

    # -------------------------------------------- #
    #  FORMAT                                      #
    # -------------------------------------------- #
    def symbol_to_pair(self, symbol: str) -> str:
        """
        Convert 'BTC' to 'BTC/USDC:USDC'
        """
        base = 'USDC'
        return f"{symbol}/{base}:{base}"

    def pair_to_symbol(self, symbol: str) -> str:
        """
        Convert 'BTC/USDC:USDC' to 'BTC'
        """
        return symbol.split('/')[0]
    
    def size_to_precision(self, symbol: str, size: float) -> float:
        pair = self.symbol_to_pair(symbol)
        size_precision = self.market[pair].contract_precision
        decimal_precision = Decimal(str(size_precision))
        rounded_size = Decimal(str(size)).quantize(decimal_precision, rounding=ROUND_DOWN)
        return float(rounded_size)
    
    # -------------------------------------------- #
    #  MARKETS                                     #
    # -------------------------------------------- #
    async def load_markets(self):
        data = await self._dex.publicPostInfo(params={"type": "metaAndAssetCtxs"})
        meta = data[0]["universe"]
        asset_info = data[1]
        resp = {}
        for i, obj in enumerate(meta):
            name = obj["name"]  # e.g. "BTC"
            mark_price = float(asset_info[i]["markPx"])
            size_decimals = int(obj["szDecimals"])
            # Use a proper tick size if available; otherwise, default to 0.01 (adjust as needed)
            if "priceDecimals" in obj:
                price_precision = 1 / (10 ** int(obj["priceDecimals"]))
            else:
                price_precision = 0.01  # default tick size
            
            item = Market(
                internal_pair=name,
                base=name,
                quote="USDC",
                price_precision=price_precision,
                contract_precision=1 / (10 ** size_decimals),
                min_contracts=1 / (10 ** size_decimals),
                coin_index=i,
                market_price=mark_price,
            )
            pair = f"{name}/USDC:USDC"
            resp[pair] = item
        self.market = resp

    def get_pair_info(self, symbol) -> str:
        pair = self.symbol_to_pair(symbol)
        if pair in self.market:
            return self.market[pair]
        else:
            return None
    # -------------------------------------------- #
    #  PRICE                                       #
    # -------------------------------------------- #
    async def get_mid_price(self, symbol: str) -> float | None:
        """
        symbol as 'BTC' not 'BTC/USDC:USDC'
        """
        data = await self._dex.publicPostInfo(params={"type": "metaAndAssetCtxs"})
        if len(data) < 2:
            return None

        universe = data[0].get("universe", [])
        asset_ctxs = data[1]

        for asset_info, asset_ctx in zip(universe, asset_ctxs):
            if asset_info.get("name") == symbol:
                mid_px_str = asset_ctx.get("midPx")
                if mid_px_str is not None:
                    return float(mid_px_str)
                return None
        return None

    async def get_best_price(self, symbol: str, side: str) -> Optional[float]:
        try:
            pair = self.symbol_to_pair(symbol)
            ob = await self._dex.fetch_order_book(pair)
            if side == 'long':
                return ob["asks"][0][0]
            elif side == 'short':
                return ob["bids"][0][0]
        except Exception as e:
            print(f"Error while fetching best price for {symbol}: {e}")
        return None
    
    async def get_last_ohlcv(self, symbol, timeframe, limit=1000) -> pd.DataFrame:
        if limit > 5000:
            limit = 5000
        ts_dict = {
            "1m": 1 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - ((limit-1) * ts_dict[timeframe])
        data = await self._dex.publicPostInfo(params={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": timeframe,
                "startTime": start_ts,
                "endTime": end_ts,
            },
        })
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['t'].astype(float), unit='ms')
        df.set_index('date', inplace=True)
        df = df[['o', 'h', 'l', 'c', 'v']].astype(float)
        df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)

        return df
    
    # -------------------------------------------------------------------------
    # Precision helpers
    # -------------------------------------------------------------------------
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        # pair = self.symbol_to_pair(symbol)
        if symbol not in self.market:
            return 0.0
        market = self.market[symbol]
        size_prec = Decimal(str(market.contract_precision))
        out = Decimal(str(amount)).quantize(size_prec, rounding=ROUND_DOWN)
        return float(out)

    def price_to_precision(self, symbol: str, price: float) -> float:
        pair = self.symbol_to_pair(symbol)
        if pair not in self.market:
            return 0.0
        market = self.market[pair]
        step = Decimal(str(market.price_precision))
        pdec = Decimal(str(price))
        if step <= 0:
            return float(price)
        # integer-div style rounding
        steps = (pdec // step) * step
        return float(steps)

    # async def get_last_ohlcv(self, symbol, timeframe, limit=1000) -> pd.DataFrame:
    #     # pair = self.symbol_to_pair(symbol)
    #     if symbol is None:
    #         return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    #     tf_map = {
    #         "1m": 60_000,
    #         "5m": 5 * 60_000,
    #         "15m": 15 * 60_000,
    #         "1h": 60 * 60_000,
    #         "2h": 2 * 60 * 60_000,
    #         "4h": 4 * 60 * 60_000,
    #         "1d": 24 * 60 * 60_000,
    #     }
    #     ms_per_candle = tf_map.get(timeframe, 60_000)
    #     end_ts = int(time.time() * 1000)
    #     start_ts = end_ts - (limit * ms_per_candle)

    #     data = await self._dex.publicPostInfo(params={
    #         "type": "candleSnapshot",
    #         "req": {
    #             "coin": symbol,
    #             "interval": timeframe,
    #             "startTime": start_ts,
    #             "endTime": end_ts,
    #         },
    #     })
    #     df = pd.DataFrame(data)
    #     if df.empty:
    #         return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    #     df["date"] = pd.to_datetime(df["t"].astype(float), unit='ms')
    #     df.set_index("date", inplace=True)
    #     df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    #     df = df[["open","high","low","close","volume"]].astype(float)
    #     df.sort_index(inplace=True)
    #     return df
    # -------------------------------------------- #
    #  POSITIONS                                   #
    # -------------------------------------------- #
    async def position_infos(self, symbol: str) -> OpenPosition:
        pair = self.symbol_to_pair(symbol)
        params = {"vaultAddress": self.public_address}
        raw = await self._dex.fetch_position(pair, params)

        # Build an OpenPosition model
        return OpenPosition(
            id=raw.get("id"),
            symbol=raw["symbol"],
            timestamp=raw.get("timestamp"),
            datetime=raw.get("datetime"),
            isolated=raw["isolated"],
            hedged=raw.get("hedged"),
            side=await self.get_position_side(symbol),
            contracts=raw["contracts"],
            contractSize=raw["contractSize"],
            entryPrice=raw["entryPrice"],
            markPrice=raw.get("markPrice"),
            notional=raw["notional"],
            leverage=raw["leverage"],
            collateral=raw.get("collateral"),
            initialMargin=raw["initialMargin"],
            maintenanceMargin=raw.get("maintenanceMargin"),
            initialMarginPercentage=raw.get("initialMarginPercentage"),
            maintenanceMarginPercentage=raw.get("maintenanceMarginPercentage"),
            unrealizedPnl=raw["unrealizedPnl"],
            liquidationPrice=raw.get("liquidationPrice"),
            marginMode=raw.get("marginMode"),
            percentage=raw["percentage"]
        )
    
    async def get_open_positions(self, pairs=[]) -> List[Position]:
        data = await self._dex.publicPostInfo(params={
            "type": "clearinghouseState",
            "user": self.public_address,
        })
        # return data
        positions_data = data["assetPositions"]
        positions = []
        for position_data in positions_data:
            position = position_data["position"]
            if self.pair_to_symbol(position["coin"]) not in pairs and len(pairs) > 0:
                continue
            type_mode = position_data["type"]
            hedge_mode = True if type_mode != "oneWay" else False
            size = float(position["szi"])
            side = "long" if size > 0 else "short"
            size = abs(size)
            usd_size = float(position["positionValue"])
            current_price = usd_size / size
            positions.append(
                Position(
                    pair=self.symbol_to_pair(position["coin"]),
                    side=side,
                    size=size,
                    usd_size=usd_size,
                    entry_price=float(position["entryPx"]),
                    current_price=current_price,
                    unrealized_pnl=float(position["unrealizedPnl"]),
                    # liquidation_price=float(position["liquidationPx"]),
                    margin_mode=position["leverage"]["type"],
                    leverage=position["leverage"]["value"],
                    hedge_mode=hedge_mode,
                )
            )

        return positions

    async def check_in_position(self, symbol: str) -> bool:
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            open_positions = await self._dex.fetch_positions()
            for position in open_positions:
                if position['symbol'] == pair:
                    position_value = float(position['info']['position']['positionValue'])
                    in_position = position_value > 0
                    return in_position
            return False
        except Exception as e:
            print(f"Error checking position for {symbol}: {e}")
            return False

    async def get_position_side(self, symbol: str) -> Optional[str]:
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            pos = await self._dex.fetch_position(pair, params)
            szi = float(pos['info']['position']['szi'])
            if szi > 0:
                return 'long'
            elif szi < 0:
                return 'short'
            else:
                return 'flat'
        except Exception as e:
            print(f"Error getting position side for {symbol}: {e}")
            return None

    async def invert_side(self, symbol: str) -> Optional[str]:
        try:
            current_side = await self.get_position_side(symbol)
            if current_side == 'long':
                return 'sell'
            elif current_side == 'short':
                return 'buy'
        except Exception as e:
            print(f"Error getting order side for {symbol}: {e}")
        return None

    # -------------------------------------------- #
    #  ORDERS                                      #
    # -------------------------------------------- #
    async def get_open_orders(self, symbol: str) -> list[OpenOrder]:
        pair = self.symbol_to_pair(symbol)
        raw_orders = await self._dex.fetch_open_orders(pair)

        if not raw_orders:
            print(f"No open orders found for {symbol}")
            return []
        else:
            print(f"Found {len(raw_orders)} open orders for {symbol}")

        open_orders = []
        for order_data in raw_orders:
            open_order = OpenOrder(
                id=order_data.get("id"),
                clientOrderId=order_data.get("clientOrderId"),
                timestamp=order_data.get("timestamp"),
                datetime=order_data.get("datetime"),
                lastTradeTimestamp=order_data.get("lastTradeTimestamp"),
                lastUpdateTimestamp=order_data.get("lastUpdateTimestamp"),
                symbol=order_data["symbol"],
                type=order_data["type"],
                timeInForce=order_data.get("timeInForce") or "GTC",
                postOnly=bool(order_data.get("postOnly")) if order_data.get("postOnly") is not None else False,
                reduceOnly=order_data["reduceOnly"],
                side=order_data["side"],
                price=float(order_data["price"]),
                triggerPrice=order_data.get("triggerPrice"),
                amount=float(order_data["amount"]),
                cost=order_data.get("cost"),
                average=order_data.get("average"),
                filled=order_data.get("filled"),
                remaining=order_data.get("remaining"),
                status=order_data.get("status"),
                fee=order_data.get("fee"),
                trades=order_data.get("trades", []),
                fees=order_data.get("fees", []),
                stopPrice=order_data.get("stopPrice"),
                takeProfitPrice=order_data.get("takeProfitPrice"),
                stopLossPrice=order_data.get("stopLossPrice"),
            )
            open_orders.append(open_order)
        return open_orders
        
    async def cancel_orders(self, symbol: str):
        try:
            orders = await self.get_open_orders(symbol)
            params = {"vaultAddress": self.public_address}
            if orders:
                for order in orders:
                    pair = self.symbol_to_pair(symbol)
                    await self._dex.cancel_order(order.id, pair, params)
                print(f"Successfully cancelled {len(orders)} order(s) for {symbol}")
            else:
                print(f"Didn't find any order for {symbol}")
        except Exception as e:
            print(f"Error cancelling order(s) for {symbol}: {e}")

    async def set_margin_mode_and_leverage(self, symbol, margin_mode, leverage):
        if margin_mode not in ["cross", "isolated"]:
            raise Exception("Margin mode must be either 'cross' or 'isolated'")
        pair = self.symbol_to_pair(symbol)
        asset_index = self.market[pair].coin_index
        try:
            nonce = int(time.time() * 1000)
            req_body = {}
            action = {
                "type": "updateLeverage",
                "asset": asset_index,
                "isCross": margin_mode == "cross",
                "leverage": leverage,
            }
            signature = self._dex.sign_l1_action(action, nonce)
            req_body["action"] = action
            req_body["nonce"] = nonce
            req_body["signature"] = signature
            await self._dex.private_post_exchange(params=req_body)
        except Exception as e:
            raise e

        return Info(
            success=True,
            message=f"Margin mode and leverage set to {margin_mode} and {leverage}x",
        )

    async def get_order_by_id(self, order_id) -> OpenOrder:
        order_id = int(order_id)
        data = await self._dex.publicPostInfo(params={
            "user": self.public_address,
            "type": "orderStatus",
            "oid": order_id,
        })
        order = data["order"]["order"]
        side_map = {
            "A": "sell",
            "B": "buy",
        }
        return OpenOrder(
                id=order.get("id"),
                clientOrderId=order.get("clientOrderId"),
                timestamp=order.get("timestamp"),
                datetime=order.get("datetime"),
                lastTradeTimestamp=order.get("lastTradeTimestamp"),
                lastUpdateTimestamp=order.get("lastUpdateTimestamp"),
                symbol=order["coin"],
                type=order["orderType"].lower(),
                timeInForce=order.get("timeInForce") or "GTC",
                postOnly=bool(order.get("postOnly")) if order.get("postOnly") is not None else False,
                reduceOnly=order["reduceOnly"],
                side=order["side"],
                price=float(order["limitPx"]),
                triggerPrice=order.get("triggerPrice"),
                amount=float(order["origSz"]),
                cost=order.get("cost"),
                average=order.get("average"),
                filled=order.get("filled"),
                remaining=order.get("remaining"),
                status=order.get("status"),
                fee=order.get("fee"),
                trades=order.get("trades", []),
                fees=order.get("fees", []),
                stopPrice=order.get("stopPrice"),
                takeProfitPrice=order.get("takeProfitPrice"),
                stopLossPrice=order.get("stopLossPrice"),
            )

    # -------------------------------------------- #
    #  OPEN POSITIONS (Create Orders)             #
    # -------------------------------------------- #
    async def market_buy(self, symbol: str, amount: float):
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            price = await self.get_mid_price(symbol)
            order = await self._dex.create_order(pair, 'market', 'buy', amount, price, params)
            print(f"Market buy order successfully placed")
            print(order)
        except Exception as e:
            print(f"Error placing market buy order for {symbol}: {e}")

    async def market_sell(self, symbol: str, amount: float):
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            price = await self.get_mid_price(symbol)
            order = await self._dex.create_order(pair, 'market', 'sell', amount, price, params)
            print(f"Market sell order successfully placed")
            print(order)
        except Exception as e:
            print(f"Error placing market sell order for {symbol}: {e}")

    async def limit_buy(self, symbol: str, amount: float, price: float) -> OpenOrder:
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            order = await self._dex.create_order(pair, 'limit', 'buy', amount, price, params)
            order_id = order['id']
            order = await self.get_order_by_id(order_id)
            order.price = float(price)
            print(f"Limit buy order successfully placed")
            return order  # Return the order object here
        except Exception as e:
            print(f"Error placing limit buy order for {symbol}: {e}")
            return None  # Optionally return None on error

    async def limit_sell(self, symbol: str, amount: float, price: float) -> OpenOrder:
        try:
            params = {"vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            order = await self._dex.create_order(pair, 'limit', 'sell', amount, price, params)
            order_id = order['id']
            order = await self.get_order_by_id(order_id)
            order.price = float(price)
            print(f"Limit sell order successfully placed")
            return order  # Return the order object here
        except Exception as e:
            print(f"Error placing limit sell order for {symbol}: {e}")
            return None  # Optionally return None on error

    async def place_order(
        self,
        symbol,
        side,
        price,
        size,
        type="limit",
        reduce=False,
        error=True,
        market_max_spread=0.1,
    ) -> Order:
        # Convert symbol to pair once
        pair = self.symbol_to_pair(symbol)
        if price is None:
            price = self.market[pair].market_price
        try:
            asset_index = self.market[pair].coin_index
            nonce = int(time.time() * 1000)
            is_buy = side == "buy"
            req_body = {}
            if type == "market":
                if side == "buy":
                    price = price * (1 + market_max_spread)
                else:
                    price = price * (1 - market_max_spread)

            print(number_to_str(self.price_to_precision(symbol, price)))
            action = {
                "type": "order",
                "orders": [{
                    "a": asset_index,
                    "b": is_buy,
                    "p": number_to_str(self.price_to_precision(symbol, price)),
                    "s": number_to_str(self.size_to_precision(symbol, size)),
                    "r": reduce,
                    "t": {"limit": {"tif": "Gtc"}}
                }],
                "grouping": "na",
                "brokerCode": 1,
            }
            signature = self._dex.sign_l1_action(action, nonce)
            req_body["action"] = action
            req_body["nonce"] = nonce
            req_body["signature"] = signature
            resp = await self._dex.private_post_exchange(params=req_body)
            
            order_resp = resp["response"]["data"]["statuses"][0]
            order_key = list(order_resp.keys())[0]
            order_id = resp["response"]["data"]["statuses"][0][order_key]["oid"]

            order = await self.get_order_by_id(order_id)

            if order_key == "filled":
                order_price = resp["response"]["data"]["statuses"][0][order_key]["avgPx"]
                order.price = float(order_price)
            
            return order
        except Exception as e:
            if error:
                raise e
            else:
                print(e)
                return None
        
    # -------------------------------------------- #
    #  CLOSE POSITIONS                             #
    # -------------------------------------------- #
    async def market_close(self, symbol: str):
        try:
            params = {"reduceOnly": True, "vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = pos.contracts
            price = await self.get_mid_price(symbol)
            side = await self.invert_side(symbol)
            order = await self._dex.create_order(pair, 'market', side, size, price, params)
            print(f"Market close order successfully placed for {symbol}")
            print(order)
        except Exception as e:
            print(f"Error placing market close order for {symbol}: {e}")

    async def limit_close(self, symbol: str):
        try:
            params = {"reduceOnly": True, "vaultAddress": self.public_address}
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = pos.contracts
            curr_side = await self.get_position_side(symbol)
            price = await self.get_best_price(symbol, curr_side)
            side = await self.invert_side(symbol)
            order = await self._dex.create_order(pair, 'limit', side, size, price, params)
            order_id = order['id']
            order = await self.get_order_by_id(order_id)
            order.price = float(price)
            print(f"Limit {side} order successfully placed to close {symbol} {curr_side} position")
            return order
        except Exception as e:
            print(f"Error placing limit close order for {symbol}: {e}")

    async def close_all_positions(self):
        try:
            all_pos = await self._dex.fetch_positions()
            for pos in all_pos:
                params = {"reduceOnly": True, "vaultAddress": self.public_address}
                pos_symbol = pos['info']['position']['coin']
                pos_pair = self.symbol_to_pair(pos_symbol)
                pos_side = await self.get_position_side(pos_symbol)
                pos_size = abs(float(pos['info']['position']['szi']))
                close_price = await self.get_best_price(pos_pair, pos_side)
                print(f"Found position for {pos_pair}, side is {pos_side}, size is {pos_size}")
                await self._dex.create_order(pos_pair, 'market', pos_side, pos_size, close_price, params)
        except Exception as e:
            print(f"Error closing all positions: {e}")

    # -------------------------------------------- #
    #  TP/SL                                       #
    # -------------------------------------------- #
    async def place_tp_market(self, symbol: str, side: str, tp_price: float):
        try:
            params = {
                "takeProfitPrice": tp_price,
                "reduceOnly": True,
                "vaultAddress": self.public_address,
            }
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = pos.contracts
            price = await self.get_mid_price(symbol)
            await self._dex.create_order(pair, 'market', side, size, price, params)
        except Exception as e:
            print(f"Error place TP Market for {symbol}: {e}")

    async def place_sl_market(self, symbol: str, side: str, sl_price: float):
        try:
            params = {
                "stopLossPrice": sl_price,
                "reduceOnly": True,
                "vaultAddress": self.public_address
            }
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = pos.contracts
            price = await self.get_mid_price(symbol)
            await self._dex.create_order(pair, 'market', side, size, price, params)
        except Exception as e:
            print(f"Error place TP Market for {symbol}: {e}")

    async def place_tp_limit(self, symbol: str, side: str, size, tp_price: float):
        try:
            params = {
                "takeProfitPrice": tp_price,
                "reduceOnly": True,
                "vaultAddress": self.public_address
            }
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = abs(pos.contracts)
            price = await self.get_mid_price(symbol)
            await self._dex.create_order(pair, 'limit', side, size, price, params)
        except Exception as e:
            print(f"Error place TP Limit for {symbol}: {e}")

    async def place_sl_limit(self, symbol: str, side: str, sl_price: float):
        try:
            params = {
                "stopLossPrice": sl_price,
                "reduceOnly": True,
                "vaultAddress": self.public_address
            }
            pair = self.symbol_to_pair(symbol)
            pos = await self.position_infos(symbol)
            size = pos.contracts
            price = await self.get_mid_price(symbol)
            await self._dex.create_order(pair, 'limit', side, size, price, params)
        except Exception as e:
            print(f"Error place SL Limit for {symbol}: {e}")

    # -------------------------------------------- #
    #  CLEANUP (close ccxt sessions if needed)     #
    # -------------------------------------------- #
    async def close(self):
        """
        Explicitly close the ccxt session if you need to shut down cleanly.
        """
        await self._dex.close()