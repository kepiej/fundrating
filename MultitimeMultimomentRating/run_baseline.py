import logging
from pathlib import Path

import bt
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FixedAssetSelect(bt.Algo):
    def __init__(self, assetK: int):
        super().__init__()
        self.assetK = assetK

    def __call__(self, target: bt.Strategy):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        target.temp["weights"] = {
            el: 1 if k == self.assetK else 0 for k, el in enumerate(selected)
        }

        return True


def getBackTestList(
    price_df: pd.DataFrame,
    assetK: int,
    max_window_years: int = 5,
    **kwargs,
) -> list[bt.Backtest]:
    index_startdate = (
        price_df.index[1]
        + pd.DateOffset(years=max_window_years)
        - pd.DateOffset(months=1)
    )
    logger.info(f"Start date of first rebalance: {index_startdate}")

    # Vary the rebalancing time by generating a new strategy for every rebalance. This eliminates the effect of timing.
    testlist = []
    nrebalances = (
        (price_df.index.max().to_period("M") - price_df.index.min().to_period("M")).n
        - (2 * max_window_years * 12)
    )
    logger.info(f"Number of rebalances is: {nrebalances}")
    for k in range(nrebalances):
        index_rebdate = index_startdate + pd.DateOffset(months=k)
        s = bt.Strategy(
            name=f"Rebalance_{k}",
            algos=[
                bt.algos.RunAfterDate(date=index_rebdate - pd.DateOffset(months=1)),
                bt.algos.RunOnce(),
                bt.algos.SelectAll(),
                FixedAssetSelect(assetK=assetK),
                bt.algos.Rebalance(),
            ],
        )
        testlist.append(bt.Backtest(s, price_df, additional_data=kwargs))

    return testlist


if __name__ == "__main__":
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    DATA_PATH: Path = Path.cwd()

    logger.info(f"Reading price data from {DATA_PATH / 'prices.parquet'}...")
    data_prices = pd.read_parquet(DATA_PATH / "prices.parquet")
    logger.info(f"Reading dividend data from {DATA_PATH / 'dividends.parquet'}...")
    data_dividends = pd.read_parquet(DATA_PATH / "dividends.parquet")
    # Set NaN values to 0
    data_prices = data_prices.fillna(0.0).sort_index()
    data_dividends = data_dividends.fillna(0.0).sort_index()

    example = bt.run(
        *getBackTestList(
            data_prices,
            assetK=0,
            max_window_years=5,
        )
    )
    totalres = pd.DataFrame(
        data=pd.NA,
        index=pd.MultiIndex.from_product([example.stats.index, range(data_prices.shape[1])]),
        columns=example.stats.columns,
    )
    del example
    for assetK in tqdm(range(data_prices.shape[1])):
        testlist = getBackTestList(
            data_prices,
            assetK=assetK,
            max_window_years=5,
        )
        try:
            res = bt.run(*testlist)
            totalres.loc[(slice(None), assetK), :] = res.stats.values

        except ValueError as valerr:
            logger.error(valerr)
            continue
        finally:
            del res
            del testlist
    
    totalres.to_excel(DATA_PATH / "Baseline_FixedFundSelect.xlsx")
    #totalres.to_parquet(DATA_PATH / "Baseline_FixedFundSelect.parquet")