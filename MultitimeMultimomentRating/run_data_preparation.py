from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    # Path to dropbox folder where raw data is present in subfolders 'Multimoment, multitime fund ratings with robust moment statistics' and 'Multi-horizon Robust Moment_Sample data'
    # Adjust this path to point to your Dropbox folder on your pc
    DROPBOXFOLDER_PATH: Path = Path('/home/kepiej/Dropbox/ATarnaud/')
    # Path where processed data should be written to
    PROCESSED_DATA_PATH: Path = Path.cwd()

    if (PROCESSED_DATA_PATH / 'New series of NAVs & dividends_NAVS.parquet').exists():
        logger.info(f'"New series of NAVs & dividends_NAVS.parquet" found at {PROCESSED_DATA_PATH}! Reading parquet file...')
        navs_df = pd.read_parquet(PROCESSED_DATA_PATH / 'New series of NAVs & dividends_NAVS.parquet')
    else:
        logger.info(f'No "New series of NAVs & dividends_NAVS.parquet" file found at {PROCESSED_DATA_PATH}! Reading Excel file...')
        navs_df = pd.read_excel(DROPBOXFOLDER_PATH / 'Multimoment, multitime fund ratings with robust moment statistics' / 'New series of NAVs & dividends.xlsx', sheet_name='NAVs')
        logger.info(f'Writing Parquet file of raw data to {PROCESSED_DATA_PATH / 'New series of NAVs & dividends_NAVS.parquet'} for future processing...')
        navs_df.astype({'Distributions Default EUR // NAME': pd.StringDtype(), 'Lipper ID': pd.StringDtype()}).to_parquet(PROCESSED_DATA_PATH / 'New series of NAVs & dividends_NAVS.parquet')

    sample_df = pd.read_excel(DROPBOXFOLDER_PATH / 'Multi-horizon Robust Moment_Sample data' / '2-Sample-EquityFunds-693_15May.xlsx', sheet_name='SimpleFormat', usecols=['Lipper ID'])
    sel_fundsid = sample_df['Lipper ID'].astype(str).to_list()
    logger.info(f"Number of selected funds: {len(sel_fundsid)}")

    subset_navs_df = navs_df[navs_df['Lipper ID'].isin(sel_fundsid)]
    prices = subset_navs_df[subset_navs_df.columns[2:]].T
    prices.index = pd.to_datetime(prices.index)

    logger.info(f'Writing selected subset of data to {PROCESSED_DATA_PATH / 'prices.parquet'}')
    prices.to_parquet(PROCESSED_DATA_PATH / 'prices.parquet')
