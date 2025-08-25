import asyncio
import streamlit as st
import pandas as pd
from get_portfolio_extended import get_extended_balance, get_extended_positions
from get_portfolio_lighter import get_lighter_balance, get_lighter_positions
from get_portfolio_drift import get_drift_balance, get_drift_positions


def get_balances():
    extended_balance = get_extended_balance()
    lighter_balance = asyncio.run(get_lighter_balance())
    drift_balance = asyncio.run(get_drift_balance())
    # Create DataFrame from list of dicts
    df = pd.DataFrame([extended_balance, lighter_balance, drift_balance])

    # Add total row
    total = pd.DataFrame({
        'exchange': ['Total'],
        'balance': [df['balance'].sum()],
        'equity': [df['equity'].sum()],
        'unrealized_pnl': [df['unrealized_pnl'].sum()]
    })

    df_total = pd.concat([df, total], ignore_index=True)

    return df_total[['exchange', 'equity']]


def get_positions():
    extended_positions = get_extended_positions()
    lighter_positions = asyncio.run(get_lighter_positions())
    drift_positions = asyncio.run(get_drift_positions())
    print(extended_positions)
    print(lighter_positions)
    print(drift_positions)
    drift_balance = {
        'exchange': 'drift',
        'balance': 5000
    }

    hyperliquid_balance = {
        'exchange': 'hyperliquid',
        'balance': 0
    }

    # Create DataFrame from list of dicts
    df = pd.DataFrame(extended_positions + lighter_positions + drift_positions)

    # # # Add total row
    # total = pd.DataFrame({
    #     'exchange': ['Total'],
    #     'balance': [df['balance'].sum()]
    # })
    # df_total = pd.concat([df, total], ignore_index=True)

    return df


def main():
    st.title("Portfolio Overview")

    st.header("Balances")
    df_balances = get_balances()
    st.table(df_balances)
    #
    st.header("Positions")
    df_positions = get_positions()
    st.table(df_positions)


if __name__ == "__main__":
    main()