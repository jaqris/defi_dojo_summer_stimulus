import os
import requests
import logging
import asyncio
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from driftpy.drift_client import DriftClient
from typing import Dict
from driftpy.math.margin import MarginCategory

logging.basicConfig(level=logging.INFO)


BASE_URL = "https://data.api.drift.trade/"

SOLANA_RPC_API_KEY = os.getenv("SOLANA_RPC_API_KEY")
RPC_URL = f'https://mainnet.helius-rpc.com/?api-key={SOLANA_RPC_API_KEY}'

KEYPAIR = Keypair()

PUBLIC_KEY = os.getenv("SOLANA_COMP_PUBLIC_KEY")
PUBLIC_KEY = Pubkey.from_string(PUBLIC_KEY)
DRIFT_ACCOUNT_ID = os.getenv("SOLANA_COMP_DRIFT_ACCOUNT_ID")


def fetch_market_index_to_symbol_map():
    contracts_url = f"{BASE_URL}/contracts"
    resp = requests.get(contracts_url)
    resp.raise_for_status()
    contracts = resp.json()

    mapping: Dict[int, str] = {}

    for item in contracts['contracts']:
        # The fields might be named differently; adjust as needed
        market_index = item.get("contract_index")
        symbol = item.get("base_currency") or item.get("symbol") or item.get("market")

        if market_index is None or symbol is None:
            continue

        try:
            market_index = int(market_index)
        except (ValueError, TypeError):
            continue

        mapping[market_index] = symbol

    return mapping


async def get_drift_balance():
    connection = AsyncClient(RPC_URL)
    drift_client = DriftClient(
        connection,
        wallet=KEYPAIR,
        env="mainnet",
        perp_market_indexes=[0, 75, 72, 66, 6,42],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    drift_user = drift_client.get_user()


    total_collateral = drift_user.get_total_collateral(MarginCategory.MAINTENANCE)
    free_collateral = drift_user.get_free_collateral(MarginCategory.MAINTENANCE)
    unrealized_pnl = drift_user.get_unrealized_pnl(with_funding=True)
    # pnl = drift_user.get_unrealized_funding_pnl()
    equity = total_collateral / 10**6 + unrealized_pnl / 10**6
    print(total_collateral)
    print(free_collateral)
    print(unrealized_pnl)
    active_positions = drift_user.get_active_perp_positions()
    notional_exposure = 0
    for p in active_positions:
        notional_exposure = notional_exposure + abs(p.quote_asset_amount / 10**6)
        # positions.append({
        #     "market_index": p.market_index,
        #     "base_asset_amount": p.base_asset_amount / 10**9,
        #     "quote_asset_amount": p.quote_asset_amount / 10**6,
        #     "price": (p.quote_asset_amount / 10**6) / abs(p.base_asset_amount / 10**9) if p.base_asset_amount != 0 else 0
        # })
    print(total_collateral)
    print(unrealized_pnl)

    margin_requirement = total_collateral - free_collateral
    health_ratio = (1 - margin_requirement / total_collateral) * 100

    balance = {
        "exchange": "drift",
        # "balance": total_collateral / 10**6,
        "equity": equity,
        # "unrealized_pnl": unrealized_pnl / 10**6
        "notional_exposure": notional_exposure,
        "leverage": str(f"{round(notional_exposure / equity, 2)}x"),
        "health_ratio": str(f"{round(health_ratio)}%")
    }

    return balance

async def get_leverage():
    connection = AsyncClient(RPC_URL)
    drift_client = DriftClient(
        connection,
        wallet=KEYPAIR,
        env="mainnet",
        perp_market_indexes=[0, 75, 72, 66, 6, 42],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    drift_user = drift_client.get_user()
    leverage = drift_user.get_leverage(include_open_orders=False)
    print(leverage)
    margin_requirement = drift_user.get_margin_requirement(margin_category="Initial")
    print(margin_requirement)


async def get_drift_positions():
    market_mappings = fetch_market_index_to_symbol_map()
    connection = AsyncClient(RPC_URL)
    drift_client = DriftClient(
        connection,
        wallet=KEYPAIR,
        env="mainnet",
        perp_market_indexes=[0, 75, 72, 66, 6, 42],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    drift_user = drift_client.get_user()

    account = drift_user.get_user_account()

    position_dicts = []
    for pos in account.perp_positions:

        if pos.base_asset_amount != 0:
            pos_dict = {
                "exchange": "drift",
                "symbol": market_mappings[pos.market_index],
                "side": "long" if pos.base_asset_amount > 0 else "short",
                "base_amount": float(pos.base_asset_amount / 10 ** 9),
            }

            position_dicts.append(pos_dict)

    return position_dicts


def get_drift_collected_funding():
    market_mappings = fetch_market_index_to_symbol_map()
    funding_url = f"{BASE_URL}/user/{DRIFT_ACCOUNT_ID}/fundingPayments"

    funding_records = []
    next_page = None

    while True:
        # add nextPage param if present
        url = funding_url if not next_page else f"{funding_url}?page={next_page}"
        resp = requests.get(url)
        resp.raise_for_status()
        results = resp.json()

        for f in results.get("records", []):
            funding_record = {
                "exchange": "drift",
                "symbol": market_mappings.get(f.get("marketIndex")),
                "timestamp": f.get("ts") * 1000,
                "funding": float(f.get("fundingPayment", 0.0)),
                "user_address": f.get("user").lower(),
            }
            funding_records.append(funding_record)

        # check if there’s another page
        next_page = results.get("meta", {}).get("nextPage")
        if not next_page:
            break

    return funding_records


def get_drift_order_history():
    trades_url = f"{BASE_URL}/user/{DRIFT_ACCOUNT_ID}/trades"
    # trades_url = "https://data.api.drift.trade/user/27AcXXxJ1Hg9uPQAnxT8ZVCZDcdhZWgLSZBsmaXdukxF/trades?page=eyJwayI6IlVTRVIjMjdBY1hYeEoxSGc5dVBRQW54VDhaVkNaRGNkaFpXZ0xTWkJzbWFYZHVreEYiLCJzayI6IlRSQURFI1RTIzE3NTcxNjIyMjUjU0xPVCMzNjUwMzYwOTcjU0lHIzJ0N0F0d281NzlUd1oyb24xNWNlVDhVYURzQ1czbmh2MnA1UjJQTGFiQURUTHp2a3ZOczlzeTl6M3ltaVZjOFBGeDdIaUxHeGZIZUNrUVFHV0oxRjZIbmYjSU5ERVgjMDAwMDAifQ%3D%3D"
    resp = requests.get(trades_url)
    resp.raise_for_status()
    results = resp.json()
    trades = []

    for t in results['records']:

        if t.get("user").lower() == t.get("taker").lower():  # Taker
            type = 'market'
            if t.get("takerOrderDirection", '').lower() == 'long':
                side = 'buy'
            else:
                side = 'sell'
            fee = float(t.get("takerFee", 0.0))
        else:  # Maker
            type = 'limit'
            if t.get("takerOrderDirection", '').lower() == 'long':
                side = 'sell'
            else:
                side = 'buy'
            fee = float(t.get("makerFee", 0.0))

        trade = {
            "exchange": "drift",
            "symbol": t.get("symbol", "").split("-")[0] if "-" in t.get("symbol", "") else t.get("symbol§", ""),
            "trade_id": t.get("txSig") + "_" + str(t.get("txSigIndex")),
            "side": side,
            "type": type,
            "price":  float(t.get("quoteAssetAmountFilled", 0.0)) / float(t.get("baseAssetAmountFilled", 1.0)),
            "filled_quantity": float(t.get("baseAssetAmountFilled", 0.0)),
            "fee": fee,
            "timestamp": t.get("ts") * 1000,
            "user_address": t.get("user").lower(),
        }
        trades.append(trade)

    return trades


async def margin():
    connection = AsyncClient(RPC_URL)
    drift_client = DriftClient(
        connection,
        wallet=KEYPAIR,
        env="mainnet",
        perp_market_indexes=[0, 75, 72, 66, 6, 42],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    from driftpy.math.margin import MarginCategory
    drift_user = drift_client.get_user()
    print(drift_user.get_active_perp_positions())

    return active_positions
    total_collateral = drift_user.get_total_collateral(MarginCategory.MAINTENANCE)
    free_collateral = drift_user.get_free_collateral(MarginCategory.MAINTENANCE)
    margin_requirement = total_collateral - free_collateral

    health_ratio =  (1 - margin_requirement / total_collateral) * 100
    print(f"Total collateral: {total_collateral / 10**6} USDC")
    print(f"Free collateral: {free_collateral / 10**6} USDC")
    print(f"Margin requirement: {margin_requirement / 10**6} USDC")
    print(f"Health ratio: {round(health_ratio)}%")

if __name__ == '__main__':
    positions = asyncio.run(get_drift_balance())
    print(positions)
