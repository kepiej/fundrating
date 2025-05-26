import datetime
import json
import re
from abc import ABC
from typing import Final, List, Optional, Protocol, Self
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

class FundProto(Protocol):

    def get_holdings(self) -> pd.DataFrame:
        ...

class iSharesABC(ABC):
    _holdingurl: Final[str]
    _columns: Final[List[str]]
    _fund_inception: Final[pd.Timestamp]
    _fund_ticker: Final[str]

    @classmethod
    def get_holdings(cls, date: pd.Timestamp) -> pd.DataFrame:
        if date < cls._fund_inception:
            raise ValueError("date is older than funds existence!")

        cururl: str = cls._holdingurl.replace("{histdate}", date.strftime("%Y%m%d"))
        response = requests.get(cururl).text
        holdingsdata = json.loads(response[1:])["aaData"]
        holdingsdf = pd.DataFrame(holdingsdata)
        if not holdingsdf.empty:
            holdingsdf.columns = cls._columns + [
                *range(holdingsdf.shape[1] - len(cls._columns))
            ]
        else:
            holdingsdf.columns = cls._columns
        return holdingsdf

    @property
    def ticker(self) -> str:
        return self._fund_ticker

    @property
    def inception_date(self) -> pd.Timestamp:
        return self._fund_inception


class iSharesSP500(iSharesABC):
    _holdingurl: Final[str] = (
        "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?tab=all&fileType=json&asOfDate={histdate}"
    )
    _columns: Final[List[str]] = [
        "Ticker",
        "Name",
        "Sector",
        "Asset Class",
        "Market Value",
        "Weight",
        "Notional Value",
        "Quantity",
        "CUSIP",
        "ISIN",
        "SEDOL",
        "Accrual Date",
    ]
    _fund_inception: Final[pd.Timestamp] = pd.Timestamp(year=2000, month=5, day=15)
    _fund_ticker: Final[str] = "IVV"


class iSharesNASDAQ100(iSharesABC):
    _holdingurl: Final[str] = (
        "https://www.ishares.com/nl/particuliere-belegger/nl/producten/253741/ishares-nasdaq-100-ucits-etf/1497735778849.ajax?tab=all&fileType=json&asOfDate={histdate}"
    )
    _columns: Final[List[str]] = [
        "Ticker",
        "Name",
        "Sector",
        "Asset Class",
        "Market Value",
        "Weight",
        "Notional Value",
        "Quantity",
        "ISIN",
    ]
    _fund_inception: Final[pd.Timestamp] = pd.Timestamp(year=2010, month=1, day=26)
    _fund_ticker: Final[str] = "CNDX"


class IndexHistory:
    def __init__(self, index: iSharesABC, last_updated: Optional[pd.Timestamp] = None, constituents: Optional[pd.DataFrame] = None):
        self._index: iSharesABC = index
        if last_updated is not None:
            self._last_updated: pd.Timestamp = last_updated
        else:
            self._last_updated: pd.Timestamp = index._fund_inception
        self._constituents: pd.DataFrame = constituents

    @property
    def index_symbol(self) -> str:
        return self._index.ticker
    
    @property
    def constituents(self) -> pd.DataFrame:
        return self._constituents
        
    @classmethod
    def from_file(cls, file: Path | str) -> Self:
        metadata = pd.read_parquet(file)
        constituents = metadata['constituents']
        index = eval(metadata['name'].iloc[0])()
        assert index.ticker == metadata['symbol'].iloc[0]
        return cls(index=index, last_updated=constituents.index.max(), constituents=constituents)
    
    def to_file(self, file: Path | str) -> None:
        metadata = self._constituents.to_frame('constituents')
        metadata['name'] = self._index.__name__
        metadata['symbol'] = self._index.ticker
        metadata.to_parquet(file)

    def get_constituents(
        self,
        start: pd.Timestamp,
        end: Optional[pd.Timestamp] = pd.Timestamp.today(),
    ) -> pd.Series:
        daterange = pd.date_range(start=start, end=end, freq="BME")
        constituents = pd.Series(index=daterange, dtype="object")
        for curdate in tqdm(daterange):
            try:
                constituents[curdate] = self._index.get_holdings(curdate)["Ticker"].to_list()
            except ValueError:
                constituents[curdate] = self._index.get_holdings(curdate - pd.DateOffset(days=1))[
                "Ticker"
                ].to_list()
        return constituents
    
    def update(self) -> None:
        ''' Update constituents list up to current date. '''
        
        update_constituents = self.get_constituents(start=self._last_updated + pd.DateOffset(days=1), end=pd.Timestamp.today() - pd.offsets.BMonthEnd())
        if self._constituents is not None:
            self._constituents.loc[update_constituents.index,] = update_constituents
        else:
            self._constituents = update_constituents

# def get_constituents(
#     fund: FundProto,
#     start: datetime.date,
#     end: Optional[datetime.date] = datetime.date.today(),
# ) -> pd.Series:
#     daterange = pd.date_range(start=start, end=end, freq="BME")
#     constituents = pd.Series(index=daterange, dtype="object")
#     for curdate in tqdm(daterange):
#         try:
#             constituents[curdate] = fund.get_holdings(curdate)["Ticker"].to_list()
#         except ValueError:
#             constituents[curdate] = fund.get_holdings(curdate - pd.DateOffset(days=1))[
#                 "Ticker"
#             ].to_list()
#     return constituents


def fix_ticker(ticker: str, replacedict) -> str:
    # Original rename_table from blogpost Teddy Koker
    # Source: https://teddykoker.com/2019/05/creating-a-survivorship-bias-free-sp-500-dataset-with-python/
    rename_table = {
        "-": "LPRAX",  # BlackRock LifePath Dynamic Retirement Fund
        "8686": "AFL",  # AFLAC
        "4XS": "ESRX",  # Express Scripts Holding Company
        "AAZ": "APC",  # Anadarko Petroleum Corporation
        "AG4": "AGN",  # Allergan plc
        "BFB": "BF-B",  # Brown-Forman Corporation
        "BF.B": "BF-B",  # Brown-Forman Corporation
        "BF/B": "BF-B",  # Brown-Forman Corporation
        "BLD WI": "BLD",  # TopBuild Corp.
        "BRKB": "BRK-B",  # Berkshire Hathaway Inc.
        "CC WI": "CC",  # The Chemours Company
        "DC7": "DFS",  # Discover Financial Services
        "GGQ7": "GOOG",  # Alphabet Inc. Class C
        "HNZ": "KHC",  # The Kraft Heinz Company
        "LOM": "LMT",  # Lockheed Martin Corp.
        "LTD": "LB",  # L Brands Inc.
        "LTR": "L",  # Loews Corporation
        "MPN": "MPC",  # Marathon Petroleum Corp.
        "MWZ": "MET",  # Metlife Inc.
        "MX4A": "CME",  # CME Group Inc.
        "NCRA": "NWSA",  # News Corporation
        "NTH": "NOC",  # Northrop Grumman Crop.
        "PA9": "TRV",  # The Travelers Companies, Inc.
        "QCI": "QCOM",  # Qualcomm Inc.
        "RN7": "RF",  # Regions Financial Corp
        "SLBA": "SLB",  # Schlumberger Limited
        "SYF-W": "SYF",  # Synchrony Financial
        "SWG": "SCHW",  # The Charles Schwab Corporation
        "UAC/C": "UAA",  # Under Armour Inc Class A
        "UBSFT": "UBSFY",  # Ubisoft Entertainment
        "USX1": "X",  # United States Steel Corporation
        "UUM": "UNM",  # Unum Group
        "VISA": "V",  # Visa Inc
    }
    if ticker in rename_table:
        fix = rename_table[ticker]
    elif ticker in replacedict:
        fix = replacedict[ticker]
    else:
        fix = re.sub(r"[^A-Z]+", "", ticker)

    return fix

if __name__ == "__main__":
    ...
