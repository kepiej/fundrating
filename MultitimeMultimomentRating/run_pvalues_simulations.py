from pathlib import Path
from typing import Final
from itertools import chain
import logging

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())

def calculate_pvalues(
    baseline: pd.DataFrame, simres: pd.DataFrame, indicator: str
) -> pd.Series:
    # p-value for each rebalance period: it uses the baseline to check how extreme the statistic on the computed MVSK portfolio is
    return 1 - baseline.loc[(indicator, slice(None)), :].transform(
        lambda row: row <= simres.loc[indicator,].values, axis=1
    ).mean(axis=0)


if __name__ == "__main__":
    # RESULTS_PATH: Final[Path] = Path.cwd()
    RESULTS_PATH: Final[Path] = Path("/home/kepiej/Dropbox/ATarnaud/Multimoment, multitime fund ratings with robust moment statistics/Results/")

    baseline = pd.read_excel(
        RESULTS_PATH / "Baseline_FixedFundSelect.xlsx", index_col=[0, 1]
    )

    # simulation_files = Path.cwd().glob("TF_RA_Funds*.xlsx")
    # simfilenames = [sim_file.stem for sim_file in simulation_files]

    simfilenames = [f"{sim_file.parts[-2]}/{sim_file.stem}" for sim_file in RESULTS_PATH.glob("./*/TF_RA_*.xlsx")] # MV, MVS results
    simfilenames.extend([f"{sim_file.parts[-2]}/{sim_file.stem}" for sim_file in RESULTS_PATH.glob("./*/*/TF_RA_*.xlsx")]) # MVSK results

    sel_indicators: Final[list[str]] = [
        "total_return",
        "cagr",
        "daily_mean",
        "daily_vol",
        "daily_skew",
        "daily_kurt",
        "monthly_mean",
        "monthly_vol",
        "monthly_skew",
        "monthly_kurt",
        "yearly_mean",
        "yearly_vol",
        "yearly_skew",
        "yearly_kurt",
        "monthly_sharpe",
        "max_drawdown",
        "calmar",
        "monthly_sortino",
        "avg_drawdown",
        "avg_drawdown_days",
    ]
    pvalindex = pd.MultiIndex.from_product([simfilenames, sel_indicators])

    prefer_less_indicators = ["daily_vol", "daily_kurt", "monthly_vol", "monthly_kurt", "yearly_vol", "yearly_kurt", "avg_drawdown_days"]

    # P-values for a selection of indicators for all simulations (stored in separate Excel files)
    pvalres = pd.DataFrame(index=pvalindex, columns=baseline.columns).sort_index()
    #for sim_file in tqdm(Path.cwd().glob("TF_RA_Funds*.xlsx")):
    for sim_file in tqdm(chain(RESULTS_PATH.glob("./*/TF_RA_*.xlsx"), RESULTS_PATH.glob("./*/*/TF_RA_*.xlsx"))):
        res = pd.read_excel(sim_file, index_col=0, usecols=range(baseline.shape[1]+1))
        logger.info(f"Current file: {sim_file}")
        for indicator in sel_indicators:
            pval = calculate_pvalues(
                baseline, res, indicator
            ).values
            if indicator in prefer_less_indicators:
                pvalres.loc[(f"{sim_file.parts[-2]}/{sim_file.stem}", indicator),] = 1- pval
            else:
                pvalres.loc[(f"{sim_file.parts[-2]}/{sim_file.stem}", indicator),] = pval

    pvalres.to_excel(RESULTS_PATH / "pvalues_simulations.xlsx")
