import logging
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, List

import bt
import pandas as pd
from fundrating import (
    MVSK,
    TF_RA,
    TF_RL,
    LMoments,
    MVSKRating,
    TLMoments,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def getBackTestList(
    price_df: pd.DataFrame,
    moment_generating_func: Callable[[pd.DataFrame], pd.DataFrame],
    max_window_years: int = 5,
    getXYgXgY=TF_RA,
    nr_moments: int = 4,
    useConvex: bool = False,
    nselectassets: int = 30,
    **kwargs,
) -> List[bt.Backtest]:
    index_startdate = (
        price_df.index[1]
        + pd.DateOffset(years=max_window_years)
        - pd.DateOffset(months=1)
    )
    logger.info(f"Start date of first rebalance: {index_startdate}")

    # Vary the rebalancing time by generating a new strategy for every rebalance. This eliminates the effect of timing.
    testlist = []
    # nrebalances = price_df.shape[0] - 1 - (2*max_window_years*12) - 1
    nrebalances = (
        (price_df.index.max().to_period("M") - price_df.index.min().to_period("M")).n
        - (2 * max_window_years * 12)
        - 1
    )
    logger.info(f"Number of rebalances is: {nrebalances}")
    for k in range(nrebalances):
        index_rebdate = index_startdate + pd.DateOffset(months=k)
        s = bt.Strategy(
            name=f"{moment_generating_func.__name__}Rating_{k}",
            algos=[
                bt.algos.RunAfterDate(date=index_rebdate - pd.DateOffset(months=1)),
                bt.algos.RunOnce(),
                bt.algos.SelectAll(),
                # bt.algos.SelectHasData(lookback = pd.DateOffset(years=max_window_years)),
                MVSKRating(
                    moment_generating_func=moment_generating_func,
                    getXYgXgY=getXYgXgY,
                    nselectassets=nselectassets,
                    useConvex=useConvex,
                    max_window_years=max_window_years,
                    nr_moments=nr_moments,
                ),
                bt.algos.Rebalance(),
            ],
        )
        testlist.append(bt.Backtest(s, price_df, additional_data=kwargs))

    return testlist


def wrap_TLMoments(trim: tuple[int]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrapper function that includes the trim parameter in the string representation of the function.
    This comes in handy when saving the results to a file where the filename also includes the method used to calculate the statistical moments.
    """

    wrapTLMoments = partial(TLMoments, trim=trim)
    wrapTLMoments.__name__ = f"TLMoments{trim[0]}-{trim[1]}"
    return wrapTLMoments


def TLMoments_percentile(r: pd.DataFrame, alfa: tuple[int]) -> pd.DataFrame:
    """
    The trim = (s,t) parameter in TLMoments corresponds to removing the s smallest and t largest observations from the sample.
    It's easier to determine these by specifying percentiles of the sample you want to remove. This function takes alfa = (l, u)
    where l, u in [0, 100] and computes the corresponding number of observations to remove.
    """

    trim = (
        round(r.shape[0] * alfa[0] / 100),
        round(r.shape[0] - (r.shape[0] * alfa[1] / 100)),
    )
    return TLMoments(r, trim)


def wrap_TLMoments_percentile(
    alfa: tuple[int],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrapper function that includes the trim parameter in the string representation of the function.
    This comes in handy when saving the results to a file where the filename also includes the method used to calculate the statistical moments.
    """

    wrapTLMoments_percentile = partial(TLMoments_percentile, alfa=alfa)
    wrapTLMoments_percentile.__name__ = f"TLMoments{alfa[0]}-{alfa[1]}"
    return wrapTLMoments_percentile


if __name__ == "__main__":
    DATA_PATH: Path = Path.cwd()

    logger.info(f"Reading price data from {DATA_PATH / 'prices.parquet'}...")
    data_prices = pd.read_parquet(DATA_PATH / "prices.parquet")
    logger.info(f"Reading dividend data from {DATA_PATH / 'dividends.parquet'}...")
    data_dividends = pd.read_parquet(DATA_PATH / "dividends.parquet")
    # Set NaN values to 0
    data_prices = data_prices.fillna(0.0).sort_index()
    data_dividends = data_dividends.fillna(0.0).sort_index()

    # Number of efficient funds to select for the portfolio
    nselectassets = [10, 20, 30]

    # Convexity
    convexities = [False]  # [True, False]

    # List of moment-generating functions to use
    momentfuncs = [
        MVSK, # Classic statistical moments
        LMoments, # L-moments
        # wrap_TLMoments(
        #     trim=(1, 1)
        # ),  # Remove smallest and largest observation from data
        # wrap_TLMoments_percentile(
        #     alfa=(10, 90)
        # ),  # Keep observations that fall between the 10% and 90% percentiles
        # wrap_TLMoments_percentile(
        #     alfa=(5, 95)
        # ),  # Keep observations that fall between the 5% and 95% percentiles
        # wrap_TLMoments_percentile(
        #     alfa=(1, 99)
        # ), # Keep observations that fall between the 1% and 99% percentiles
    ]

    for nselectasset, useConvex, cur_nr_moments, cur_mom_gen_func in tqdm(
        product(nselectassets, convexities, range(2, 4), momentfuncs)
    ):
        testlist = getBackTestList(
            data_prices,
            moment_generating_func=cur_mom_gen_func,
            max_window_years=5,
            getXYgXgY=TF_RA,
            nr_moments=cur_nr_moments,
            useConvex=useConvex,
            nselectassets=nselectasset,
            dividends=data_dividends,
        )
        try:
            res = bt.run(*testlist)
            res.stats.to_excel(
                DATA_PATH
                / f"TF_RA_Funds{nselectasset}_{cur_mom_gen_func.__name__}_{cur_nr_moments}moments_{'convex' if useConvex else 'nonconvex'}.xlsx"
            )

        except ValueError as valerr:
            logger.error(valerr)
            continue
        finally:
            del res
            del testlist
