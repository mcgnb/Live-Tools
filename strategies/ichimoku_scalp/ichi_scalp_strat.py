import sys
sys.path.append("./live_strategies")

import math
import datetime
from typing import Dict, Any
import asyncio

from utils.hl_perp import PerpHyperliquid
from utils.discord_logger import DiscordLogger

from utils.custom_indicators import (
    ichimoku,
    adx,
    bollinger_bands_width,  
    ema,
    atr
)

from secret import ACCOUNTS

# --- / STRATEGY INDICATORS SETTINGS / ---
ADX_LVL = 35
BBW_LVL= 5
EMA_LENGTH = 200
ATR_TP = 3.2
ATR_SL = 2.5
SIZE = 0.99
#-----------------------------------------

SYMBOL = "BTC"
TIMEFRAME = "1h"
MARGIN_MODE = "cross"  # or "isolated"
LEVERAGE = 1
ACCOUNT_NAME = ""
SIDE = ["long", "short"]
DISCORD_WEBHOOK = ""

async def main():
    account = ACCOUNTS[ACCOUNT_NAME]
    margin_mode = MARGIN_MODE
    leverage = LEVERAGE
    exchange_leverage = math.ceil(leverage)
    dl = DiscordLogger(DISCORD_WEBHOOK)

    dex = PerpHyperliquid(
        public_api=account["public_api"],
        secret_api=account["secret_api"],
    )

    dl.log(f"--- Ichimoku Scalper | Execution started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        # 1) Load Markets
        await dex.load_markets()

        # 2) Check if we are in a position
        in_position = await dex.check_in_position(SYMBOL)

        try:
            print(f"Setting {margin_mode} x{exchange_leverage} on {SYMBOL}...")
            await dex.set_margin_mode_and_leverage(
                    SYMBOL, margin_mode, exchange_leverage
                )
        except Exception as e:
            print("error:",e)

        if not in_position:
            dl.log(f"Not in position for {SYMBOL}")

            # 1) Cancel all open orders
            await dex.cancel_orders(SYMBOL)

            # 2) Get your balance
            balance = await dex.get_balance()
            USDC_balance = balance.total
            dl.log(f"Balance: {round(USDC_balance, 2)} USDC")

            # 3) Get price data (e.g., last 500 candles)
            df = await dex.get_last_ohlcv(SYMBOL, TIMEFRAME, limit=500)
            if df.empty:
                print(f"No OHLCV data returned for {SYMBOL}. Exiting.")
                await dex.close()
                return
            
            # --------------------------------------------------------
            # Apply Indicators
            # Columns are: ["open", "high", "low", "close", "volume"]
            # --------------------------------------------------------

            # Ichimoku
            df = ichimoku(
                df,
                high_col="high",
                low_col="low",
                close_col="close",
                conversion_period=9,
                base_period=26,
                span_b_period=52,
                displacement=26
            )
            
            # ADX
            df = adx(
                df,
                high_col="high",
                low_col="low",
                close_col="close",
                period=14,          # typical default
                adx_col_name="adx"
            )

            # Bollinger Bands (BBW)
            df = bollinger_bands_width(
                df,
                close_col="close",
                period=20,
                std_dev=2.0,
                add_bands=True     
            )

            # EMA (e.g. 50-period)
            df = ema(
                df,
                close_col="close",
                period=EMA_LENGTH,
                column_name="ema200"
            )

            # --------------------------------------------------------
            # Access final values from last candle
            # Ichimoku columns: tenkan_sen, kijun_sen, span_a, span_b, chikou_span
            # ADX column: "adx", plus "+di" and "-di"
            # BBW column: "bbw", plus "bb_upper", "bb_middle", "bb_lower"
            # EMA column: "ema50"
            # --------------------------------------------------------
            tk_s = df["tenkan_sen"].iloc[-1]
            kj_s = df["kijun_sen"].iloc[-1]
            sp_a = df["span_a"].iloc[-1]
            sp_b = df["span_b"].iloc[-1]

            adx_val = df["adx"].iloc[-1]
            di_plus = df["+di"].iloc[-1]
            di_minus = df["-di"].iloc[-1]

            bbw_val = df["bbw"].iloc[-1]
            bb_up = df["bb_upper"].iloc[-1]
            bb_low = df["bb_lower"].iloc[-1]

            ema_val = df["ema200"].iloc[-1]
            latest_close = df["close"].iloc[-1]

            # --------------------------------------------------------
            # Example Strategy Logic
            # (Replace with your real TradingView logic)
            # --------------------------------------------------------
            ichimoku_long_condition = (tk_s > kj_s) and (sp_a > sp_b)
            adx_condition = (adx_val > ADX_LVL) 
            bbw_condition = (bbw_val < BBW_LVL) 
            above_ema = (latest_close > ema_val)

            is_long = ichimoku_long_condition and adx_condition and bbw_condition and above_ema

            ichimoku_short_condition = (tk_s < kj_s) and (sp_a < sp_b)
            below_ema = (latest_close < ema_val)

            is_short = ichimoku_short_condition and adx_condition and bbw_condition and below_ema

            if is_long:
                dl.log("Time to BUY (LONG)")
                last_price = df["close"].iloc[-1]
                open_size = (USDC_balance * SIZE) / last_price * leverage
                open_price = await dex.get_best_price(SYMBOL, 'long')

                try:
                    await dex.limit_buy(
                        symbol=SYMBOL,
                        amount = float(open_size),
                        price = float(open_price)
                    )

                    dl.log(f"Opened {open_size} {SYMBOL} long")

                except Exception as e:
                    await dl.send_now(f"Error opening {SYMBOL} long: {e}", level="ERROR")

            elif is_short:
                dl.log("Time to SELL (SHORT)")
                last_price = df["close"].iloc[-1]
                open_size = (USDC_balance * SIZE) / last_price * leverage
                open_price = await dex.get_best_price(SYMBOL, 'short')
                
                try:
                    await dex.limit_sell(
                        symbol=SYMBOL,
                        amount = float(open_size),
                        price = float(open_price)
                    )

                    dl.log(f"Opened {open_size} {SYMBOL} short")

                except Exception as e:
                    await dl.send_now(f"Error opening {SYMBOL} short: {e}", level="ERROR")
            else:
                print("Conditions not met to enter a position..")

        elif in_position:
            try:
                positions = await dex.get_open_positions(SYMBOL)
                long_exposition = sum([p.usd_size for p in positions if p.side == "long"])
                short_exposition = sum([p.usd_size for p in positions if p.side == "short"])
                unrealized_pnl = sum([p.unrealized_pnl for p in positions])
                dl.log("IN POSITION")
                dl.log(f"Unrealized PNL: {round(unrealized_pnl, 2)}$ | Long Exposition: {round(long_exposition, 2)}$ | Short Exposition: {round(short_exposition, 2)}$")
                dl.log(f"Current position:")
                for position in positions:
                    dl.log(f"{(position.side).upper()} {position.size} {position.pair} ~{position.usd_size}$ (+ {position.unrealized_pnl}$)")

                df = await dex.get_last_ohlcv(SYMBOL, TIMEFRAME, limit=1500)
                if df.empty:
                    print(f"No price data for {SYMBOL}. Exiting.")
                    await dex.close()
                    return
                
                await dex.cancel_orders(SYMBOL)
                
                df = atr(
                    df, 
                    high_col="high", 
                    low_col="low", 
                    close_col="close", 
                    period=14, 
                    atr_col_name="atr"
                )

                atrMultSL = ATR_SL
                atrMultTP = ATR_TP

                atr_val = df["atr"].iloc[-1]

                position_side = await dex.get_position_side(SYMBOL)  
                position_info = await dex.position_infos(SYMBOL)
                entry_price = position_info.entryPrice
                position_size = position_info.contracts  

                if position_side == "long":
                    try:
                        stop_price = entry_price - (atrMultSL * atr_val)
                        take_profit_price = entry_price + (atrMultTP * atr_val)

                        await dex.place_tp_limit(symbol=SYMBOL, side="sell", size=position_size, tp_price=take_profit_price)
                        await dex.place_sl_limit(symbol=SYMBOL, side="sell", sl_price=stop_price)

                        dl.log(f"TP order placed at {take_profit_price} | SL order placed at {stop_price}")

                    except Exception as e:
                        dl.log(f"ERROR PLACING TP/SL ORDERS for {position_side}")

                elif position_side == "short":
                    try:
                        stop_price = entry_price + (atrMultSL * atr_val)
                        take_profit_price = entry_price - (atrMultTP * atr_val)

                        await dex.place_tp_limit(symbol=SYMBOL, side="buy", size=abs(position_size), tp_price=take_profit_price)
                        await dex.place_sl_limit(symbol=SYMBOL, side="buy", sl_price=stop_price)

                        dl.log(f"TP order placed at {take_profit_price} | SL order placed at {stop_price}")

                    except Exception as e:
                        dl.log(f"ERROR PLACING TP/SL ORDERS for {position_side}")

            except Exception as e:
                dl.send_now(f"ERROR WHILE IN POSITION => {e}", level="ERROR")

        await dex.close()
        dl.log(f"--- Execution finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        await dl.send_discord_message(level="INFO")

    except Exception as e:
        dl.send_now(f"Error => {e}", level="ERROR")
        await dex.close()
        raise e

if __name__ == "__main__":
    asyncio.run(main())
