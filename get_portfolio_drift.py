import os
import requests
import asyncio
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from driftpy.drift_client import DriftClient
from typing import Dict

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
        perp_market_indexes=[0, 75],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    drift_user = drift_client.get_user()

    total_collateral = drift_user.get_total_collateral()
    # free_collateral = drift_user.get_free_collateral()
    unrealized_pnl = drift_user.get_unrealized_pnl(with_funding=True)
    # pnl = drift_user.get_unrealized_funding_pnl()

    balance = {
        "exchange": "drift",
        # "balance": total_collateral / 10**6,
        "equity": total_collateral / 10**6 + unrealized_pnl / 10**6,
        # "unrealized_pnl": unrealized_pnl / 10**6
    }

    return balance


async def get_drift_positions():
    market_mappings = fetch_market_index_to_symbol_map()
    connection = AsyncClient(RPC_URL)
    drift_client = DriftClient(
        connection,
        wallet=KEYPAIR,
        env="mainnet",
        perp_market_indexes=[0, 75],
        spot_market_indexes=[0],
        authority=PUBLIC_KEY
    )
    await drift_client.subscribe()

    drift_user = drift_client.get_user()

    account = drift_user.get_user_account()
    # print(account)

    position_dicts = []
    for pos in account.perp_positions:

        if pos.base_asset_amount != 0:
            perp_positions = drift_user.get_perp_position(pos.market_index)
            print(perp_positions)
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
    resp = requests.get(trades_url)
    resp.raise_for_status()
    results = resp.json()
    trades = []

    for t in results['records']:
        print(t)

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


if __name__ == '__main__':
    trades = get_drift_order_history()
    print(trades)
    # asyncio.run(get_drift_order_history())