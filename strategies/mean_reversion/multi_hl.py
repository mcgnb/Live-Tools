# strategy.py
import asyncio
import datetime
import pandas as pd
import ta
from typing import Dict, Any

import sys
sys.path.append("./Live-Tools")

from secret import ACCOUNTS
from utils.hl_perp import PerpHyperliquid
from utils.discord_logger import DiscordLogger

# --- Global Strategy Settings ---
TIMEFRAME = '1h'         
SRC_COL = 'close'
MA_LENGTH = 5
SIDES = ['long', 'short']


# --- Per-Coin Settings ---
PARAMS = {
    # "BTC": {
    #     "envelopes": [0.03, 0.05, 0.07],
    #     "sides": ["long", "short"],
    # },
    "ETH": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "SOL": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "XRP": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "DOGE": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "ADA": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "BNB": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "SUI": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "LTC": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
    "LINK": {
        "envelopes": [0.03, 0.05, 0.07],
        "sides": ["long", "short"],
    },
}

async def main():
    account = ACCOUNTS["multi_mr"]
    dl = DiscordLogger("")
    exchange = PerpHyperliquid(
        public_api=account["public_api"],
        secret_api=account["secret_api"],
    )

    dl.log(f"--- Multi MR | Execution Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    try:
        # 1) Load markets and cache market info for all coins
        await exchange.load_markets()
        market_info = {}
        for coin in list(PARAMS.keys()):
            info = exchange.get_pair_info(coin)
            if info is None:
                dl.send_now(f"Coin {coin} not found, removing from PARAMS.", level="ERROR")
                del PARAMS[coin]
            else:
                market_info[coin] = info
        coins = list(PARAMS.keys())

        # 2) Set margin mode & leverage (if applicable)
        for coin in coins:
            await exchange.set_margin_mode_and_leverage(coin, "cross", 1)

        # 3) Cache the common balance (once for the whole script)
        balance = await exchange.get_balance()
        usdc_balance = balance.total
        dl.log(f"Balance: {round(usdc_balance, 2)} USDC")

        # # 4) Cancel all open orders for all coins
        # for coin in coins:
        #     await exchange.cancel_orders(coin)

        # 5) Get open positions for all coins
        positions = await exchange.get_open_positions(coins)
        positions_by_coin = {p.pair: p for p in positions}

        long_exposition = sum([p.usd_size for p in positions if p.side == "long"])
        short_exposition = sum([p.usd_size for p in positions if p.side == "short"])
        unrealized_pnl = sum([p.unrealized_pnl for p in positions])
        if positions:
            dl.log(f"Unrealized PNL: {round(unrealized_pnl, 2)}$ | Long Exposition: {round(long_exposition, 2)}$ | Short Exposition: {round(short_exposition, 2)}$")
            dl.log(f"Current positions:")
            for position in positions:
                dl.log(f"{(position.side).upper()} {position.size} {position.pair} ~{position.usd_size}$ (+ {position.unrealized_pnl}$)")
        else:
            dl.log(f"No Open Positions")


        invert_side = {"long": "sell", "short": "buy"}

        # 6) Process each coin
        for coin in coins:
            envelopes = PARAMS[coin]["envelopes"]
            sides_allowed = PARAMS[coin]["sides"]
            in_position = await exchange.check_in_position(coin)
            if in_position:
                # --- In-Position Branch ---
                print(f"Already in position for {coin}...")
                open_orders = await exchange.get_open_orders(coin)
                stored_orders: Dict[str, Any] = {}
                for o in open_orders:
                    stored_orders[o.id] = {
                        'side': o.side,
                        'price': o.price,
                        'amount': o.amount,
                    }
                print(f"Storing existing orders in dictionary: {stored_orders}")

                await exchange.cancel_orders(coin)


                # Get a larger OHLCV series for re-entry calculation
                df = await exchange.get_last_ohlcv(coin, TIMEFRAME, limit=1500)

                if df.empty:
                    print(f"No price data for {coin}. Skipping.")
                    continue

                df['ma'] = ta.trend.sma_indicator(df[SRC_COL], MA_LENGTH)
                current_ma = df['ma'].iloc[-1]
                if pd.isna(current_ma):
                    dl.send_now(f"MA is NaN for {coin}. Skipping.", level="ERROR")
                    continue

                position_side = await exchange.get_position_side(coin)
                position_info = await exchange.position_infos(coin)
                position_size = position_info.contracts
                print(f"{coin} - Position side: {position_side}, Size: {position_size}")


                # Place TP order at the current MA
                if position_side == 'long':
                    await exchange.place_tp_limit(symbol=coin, side='sell', size=abs(position_size), tp_price=current_ma)
                    dl.log(f"TP order placed for {coin} at {current_ma:.4f} (sell)")
                elif position_side == 'short':
                    await exchange.place_tp_limit(symbol=coin, side='buy', size=abs(position_size), tp_price=current_ma)
                    dl.log(f"TP order placed for {coin} at {current_ma:.4f} (buy)")

                # Use cached balance (usdc_balance) for size calculations
                print(f"Using cached balance for {coin}: {usdc_balance} USDC")

                # Determine how many orders were canceled per side
                buy_count = sum(1 for info in stored_orders.values() if info['side'] == 'buy')
                sell_count = sum(1 for info in stored_orders.values() if info['side'] == 'sell')
                print(f"{coin}: Previously canceled => buy_count={buy_count}, sell_count={sell_count}")

                if position_side == 'long' and buy_count > 0:
                    desired_levels = envelopes[len(envelopes) -buy_count:]
                elif position_side == 'short' and sell_count > 0:
                    desired_levels = envelopes[len(envelopes) -sell_count:]
                else:
                    desired_levels = []

        
                tasks = []
                for e in desired_levels:
                    high_e = (1 / (1 - e)) - 1
                    low_price_raw = current_ma * (1 - e)
                    high_price_raw = current_ma * (1 + high_e)

                    low_price = round(low_price_raw, 3)
                    high_price = round(high_price_raw, 3)

                    if position_side == 'long':
                        size_low_raw = (usdc_balance / 24) / low_price
                        tasks.append(
                            exchange.limit_buy(
                                symbol=coin,
                                amount=round(size_low_raw, 2),
                                price=float(low_price),
                            )
                        )
                    elif position_side == 'short':
                        size_high_raw = (usdc_balance / 24) / high_price
                        tasks.append(
                            exchange.limit_sell(
                                symbol=coin,
                                amount=round(size_high_raw, 2),
                                price=float(high_price),
                            )
                        )
                if tasks:
                    await asyncio.gather(*tasks)
                    print(f"Re-established orders for {coin} at levels: {desired_levels}")
                else:
                    dl.send_now(f"No re-entry orders to place for {coin}.", level="ERROR")
            else:
                # --- Not in Position Branch ---
                print(f"Not in position for {coin}. Proceeding with new setup...")
                await exchange.cancel_orders(coin)
                df = await exchange.get_last_ohlcv(coin, TIMEFRAME, limit=500)
                if df.empty:
                    dl.send_now(f"No OHLCV data returned for {coin}. Skipping.", level="ERROR")
                    continue
                df['ma'] = ta.trend.sma_indicator(df[SRC_COL], MA_LENGTH)
                current_ma = df['ma'].iloc[-1]
                if pd.isna(current_ma):
                    dl.send_now(f"MA is NaN for {coin}. Skipping.", level="ERROR")
                    continue
                print(f"{coin} {TIMEFRAME} - {MA_LENGTH}-period MA: {current_ma:.4f}")
                tasks = []
                for e in envelopes:
                    high_e = (1 / (1 - e)) - 1
                    low_price_raw = current_ma * (1 - e)
                    high_price_raw = current_ma * (1 + high_e)
                    low_price_str = exchange.price_to_precision(coin, low_price_raw)
                    high_price_str = exchange.price_to_precision(coin, high_price_raw)
                    low_price = round(low_price_raw, 3)
                    high_price = round(high_price_raw, 3)
                    if low_price <= 0:
                        dl.send_now(f"WARNING: {coin} low_price {low_price_raw} => {low_price_str} is 0 or negative, skipping.", level="ERROR")
                        continue
                    if high_price <= 0:
                        dl.send_now(f"WARNING: {coin} high_price {high_price_raw} => {high_price_str} is 0 or negative, skipping.", level="ERROR")
                        continue
                    size_low_raw = (usdc_balance / 32) / low_price
                    size_high_raw = (usdc_balance / 32) / high_price
                    if 'long' in sides_allowed:
                        tasks.append(
                            exchange.limit_buy(
                                symbol=coin,
                                amount=round(size_low_raw, 2),
                                price=float(low_price),
                            )
                        )
                    if 'short' in sides_allowed:
                        tasks.append(
                            exchange.limit_sell(
                                symbol=coin,
                                amount=round(size_high_raw, 2),
                                price=float(high_price),
                            )
                        )
                if tasks:
                    await asyncio.gather(*tasks)
                    dl.log(f"All new limit orders placed for {coin} around symmetrical MA envelopes.")
                else:
                    dl.send_now(f"No orders placed for {coin}.", level="ERROR")
        # 7) End of strategy steps
        await exchange.close()
        dl.log(f"--- Execution Finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        await dl.send_discord_message(level="INFO")

    except Exception as e:
        await exchange.close()
        print(f"Error => {e}")
        raise e

# if __name__ == "__main__":
#     asyncio.run(main())

if __name__ == "__main__":
    async def runner():
        while True:
            await main()
            print("Sleeping for an hour...")
            await asyncio.sleep(3600)

    asyncio.run(runner())
