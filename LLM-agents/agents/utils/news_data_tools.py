from langchain_core.tools import tool
from typing import Annotated
from dataflows.vnstock_newsStock import get_news_stock


@tool
def get_news(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    look_back_months = 3
    return get_news_stock(symbol, look_back_months)
