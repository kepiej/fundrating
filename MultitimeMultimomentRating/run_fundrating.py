from itertools import product
from pathlib import Path
from typing import Callable
import logging

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


def getBackTestList(price_df: pd.DataFrame, moment_generating_func: Callable[[pd.DataFrame], pd.DataFrame], max_window_years: int = 5, getXYgXgY = TF_RA, nr_moments: int = 4, useConvex: bool = False, **kwargs):
    index_startdate = price_df.index[1] + pd.DateOffset(years=max_window_years) - pd.DateOffset(months=1)
    print(index_startdate)

    # Vary the rebalancing time by generating a new strategy for every rebalance. This eliminates the effect of timing.
    testlist = []
    #nrebalances = price_df.shape[0] - 1 - (2*max_window_years*12) - 1
    nrebalances = (price_df.index.max().to_period('M') - price_df.index.min().to_period('M')).n - (2*max_window_years*12) - 1
    print(f"Number of rebalances is: {nrebalances}")
    for k in range(nrebalances):
        index_rebdate = index_startdate + pd.DateOffset(months=k)
        s = bt.Strategy(name=f'MVSKRating_{k}', algos=[
            bt.algos.RunAfterDate(date=index_rebdate - pd.DateOffset(months=1)),
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            MVSKRating(moment_generating_func=moment_generating_func, getXYgXgY=getXYgXgY, nselectassets=30, useConvex=useConvex, max_window_years=max_window_years, nr_moments=nr_moments),
            bt.algos.Rebalance()
        ])
        testlist.append(bt.Backtest(s, price_df, additional_data=kwargs))
    
    return testlist


if __name__ == "__main__":
    DATA_PATH: Path = Path.cwd()

    logger.info(f'Reading price data from {DATA_PATH / "prices.parquet"}...')
    data_prices = pd.read_parquet(DATA_PATH / "prices.parquet")
    logger.info(f'Reading dividend data from {DATA_PATH / "dividends.parquet"}...')
    data_dividends = pd.read_parquet(DATA_PATH / "dividends.parquet")
    # Set NaN values to 0
    data_prices = data_prices.fillna(0.0)
    data_dividends = data_dividends.fillna(0.0)

    #for (useConvex ,cur_nr_moments, cur_mom_gen_func) in tqdm(product([False, True], range(4,5), [MVSK, LMoments])):
    for (useConvex ,cur_nr_moments, cur_mom_gen_func) in tqdm(product([False], range(4,5), [MVSK, LMoments])):
        testlist = getBackTestList(data_prices.sort_index(), moment_generating_func=cur_mom_gen_func, max_window_years=5, getXYgXgY=TF_RA, nr_moments=cur_nr_moments, useConvex=useConvex, dividends=data_dividends)
        res = bt.run(*testlist)
        res.stats.to_excel(DATA_PATH / f"TF_RA_{cur_mom_gen_func.__name__}_{cur_nr_moments}moments_{'convex' if useConvex else 'nonconvex'}.xlsx")