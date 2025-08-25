import os
import requests
import asyncio
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from driftpy.drift_client import DriftClient
from typing import Dict

CONTRACTS_URL = "https://data.api.drift.trade/contracts"

SOLANA_RPC_API_KEY = os.getenv("SOLANA_RPC_API_KEY")
RPC_URL = f'https://mainnet.helius-rpc.com/?api-key={SOLANA_RPC_API_KEY}'

KEYPAIR = Keypair()

PUBLIC_KEY = os.getenv("SOLANA_COMP_PUBLIC_KEY")
PUBLIC_KEY = Pubkey.from_string(PUBLIC_KEY)


def fetch_market_index_to_symbol_map():
    resp = requests.get(CONTRACTS_URL)
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


if __name__ == '__main__':
    asyncio.run(get_drift_balance())