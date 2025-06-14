import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Path to folder where raw data is present in subfolders
    # Adjust this path to the correct path on your own pc
    INPUTFOLDER_PATH: Path = Path(
        "/home/kepiej/Dropbox/ATarnaud/Multimoment, multitime fund ratings with robust moment statistics"
    )
    # Path where processed data should be written to
    PROCESSED_DATA_PATH: Path = Path.cwd()

    navs_df = pd.read_excel(
        INPUTFOLDER_PATH / "New series of NAVs & dividends.xlsx", sheet_name="NAVs"
    )

    divs_df = pd.read_excel(
        INPUTFOLDER_PATH / "New series of NAVs & dividends.xlsx", sheet_name="Dividends"
    )

    sample_df = pd.read_excel(
        INPUTFOLDER_PATH / "2-Sample-AddIndex.xlsx", sheet_name="shorter list"
    )
    subsample_df = sample_df[
        (sample_df["Valuation/Pricing Frequency"] == "Pricing Daily, Mon-Fri")
        & sample_df["Institutional Fund?"].isna()
        & (sample_df["Asset Type"] == "Equity")
        & (sample_df["Minimum initial investment"] < 100000)
    ]
    sel_fundsid = subsample_df["Lipper ID"].to_list()

    logger.info(f"Number of selected funds: {len(sel_fundsid)}")

    subset_navs_df = (
        navs_df[navs_df["Lipper ID"].isin(sel_fundsid)]
        .set_index("Lipper ID")
        .drop_duplicates()
    )
    prices = subset_navs_df[subset_navs_df.columns[1:]].T
    prices.index = pd.to_datetime(prices.index)

    subset_divs_df = (
        divs_df[divs_df["Lipper ID"].isin(sel_fundsid)]
        .set_index("Lipper ID")
        .drop_duplicates()
    )
    dividends = subset_divs_df[subset_divs_df.columns[1:]].T
    dividends.index = pd.to_datetime(dividends.index)

    logger.info(
        f"Writing selected subset of data to {PROCESSED_DATA_PATH / 'prices.parquet'}"
    )
    prices.to_parquet(PROCESSED_DATA_PATH / "prices.parquet")
    logger.info(
        f"Writing selected subset of data to {PROCESSED_DATA_PATH / 'dividends.parquet'}"
    )
    dividends.to_parquet(PROCESSED_DATA_PATH / "dividends.parquet")
