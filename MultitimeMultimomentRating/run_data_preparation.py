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

    navs_df = pd.read_excel(DROPBOXFOLDER_PATH / 'Multimoment, multitime fund ratings with robust moment statistics' / 'New series of NAVs & dividends.xlsx', sheet_name='NAVs')

    divs_df = pd.read_excel(DROPBOXFOLDER_PATH / 'Multimoment, multitime fund ratings with robust moment statistics' / 'New series of NAVs & dividends.xlsx', sheet_name='Dividends')
    
    sample_df = pd.read_excel(DROPBOXFOLDER_PATH / 'Multimoment, multitime fund ratings with robust moment statistics' / '2-Sample-AddIndex.xlsx', sheet_name='shorter list')
    subsample_df = sample_df[(sample_df['Valuation/Pricing Frequency'] == 'Pricing Daily, Mon-Fri') &
          sample_df['Institutional Fund?'].isna() &
          (sample_df['Asset Type'] == 'Equity') &
          (sample_df['Minimum initial investment'] < 100000)
          ]
    sel_fundsid = subsample_df['Lipper ID'].to_list()
    
    logger.info(f"Number of selected funds: {len(sel_fundsid)}")

    subset_navs_df = navs_df[navs_df['Lipper ID'].isin(sel_fundsid)].set_index('Lipper ID').drop_duplicates()
    prices = subset_navs_df[subset_navs_df.columns[1:]].T
    prices.index = pd.to_datetime(prices.index)

    subset_divs_df = divs_df[divs_df['Lipper ID'].isin(sel_fundsid)].set_index('Lipper ID').drop_duplicates()
    dividends = subset_divs_df[subset_divs_df.columns[1:]].T
    dividends.index = pd.to_datetime(dividends.index)

    logger.info(f'Writing selected subset of data to {PROCESSED_DATA_PATH / 'prices.parquet'}')
    prices.to_parquet(PROCESSED_DATA_PATH / 'prices.parquet')
    logger.info(f'Writing selected subset of data to {PROCESSED_DATA_PATH / 'dividends.parquet'}')
    dividends.to_parquet(PROCESSED_DATA_PATH / 'dividends.parquet')