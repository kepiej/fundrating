from typing import Callable, Tuple, Optional
import lmo
import numpy as np
import pandas as pd
from scipy.stats import moment
from toolkitvdw.production_light import dirDistDEAVRS, dirDistFDHVRS
import ffn
import bt


def General_Rank_Select_TWcom_HoldtoEnd(Rankres, nSelectAsset, rawreturn, index_holdstartdate, index_holdenddate, inCapital):
    #EffAssets = Rankres.index[Rankres == 1].tolist()
    PossibleSelectAsset = Rankres.index[Rankres <= nSelectAsset].tolist()
    if len(PossibleSelectAsset) <= nSelectAsset:
        SelectAsset_Equ = PossibleSelectAsset
        SelectAsset_More = []
        SelectAsset = SelectAsset_Equ
    else:
        SelectAsset_Equ = []
        SelectAsset_More = PossibleSelectAsset[0:nSelectAsset-1]
        # SelectAsset_More=PossibleSelectAsset(randperm(numel(PossibleSelectAsset),nSelectAsset));
        SelectAsset = SelectAsset_More
    
    holdreturn = rawreturn.loc[index_holdstartdate:index_holdenddate,:]
    nasset = holdreturn.shape[1]
    ewp = np.zeros((nasset, 1))
    ewp[SelectAsset, :] = 1/nSelectAsset

    # Compute the evolution of holding wealth for held to the end of entire sample period
    Asset_cumReturnRatio = (1 + holdreturn).cumprod().iloc[-1,:]
    ewpAsset_cumReturnRatio = Asset_cumReturnRatio @ ewp
    ewpAsset_cumReturn = inCapital * ewpAsset_cumReturnRatio
    TW = ewpAsset_cumReturn#.iloc[-1,0]
    
    #ewpAsset_ReturnRatio = holdreturn @ ewp
    #ewpAsset_Return = inCapital * ewpAsset_ReturnRatio

    #ewpAsset_TotReturnRatio = (1+holdreturn) @ ewp
    #ewpAsset_TotReturn = inCapital * ewpAsset_TotReturnRatio

    return PossibleSelectAsset, SelectAsset_Equ, SelectAsset_More, SelectAsset, TW


def RApreferences(XOBS: np.ndarray, YOBS: np.ndarray) -> Tuple[np.ndarray]:
    # Sets direction vector according to Risk Averse preferences
    gX: np.ndarray = -np.abs(XOBS)
    gY: np.ndarray = np.abs(YOBS)
    return gX, gY


def RLpreferences(XOBS: np.ndarray, YOBS: np.ndarray) -> Tuple[np.ndarray]:
    # Sets direction vector according to Risk Loving preferences
    gX: np.ndarray = np.abs(XOBS)
    gY: np.ndarray = np.abs(YOBS)
    return gX, gY


def splitMomentsToXY(moments_df: pd.DataFrame) -> Tuple[np.ndarray]:
    # Split dataframe of statistical moments into a X and Y matrix for use in efficiency analysis
    # Even moments are assigned to inputs and odd moments to outputs
    # Note: Pandas dataframe indexes start at 0!
    X: np.ndarray = moments_df.iloc[:, 1::2].to_numpy()
    Y: np.ndarray = moments_df.iloc[:, 0::2].to_numpy()
    return X, Y


def OFsplitMomentsToXY(moments_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Split dataframe of statistical moments into a X and Y matrix for use in efficiency analysis
    # All moments are assigned to the outputs and the inputs are set to a vector of zeros
    X: np.ndarray = np.zeros(shape=(moments_df.shape[0], 1))
    Y: np.ndarray = moments_df.to_numpy()
    return X, Y


def TF_RA(moments_df: pd.DataFrame) -> Tuple[np.ndarray]:
    # Converts a dataframe of statistical moments to inputs and outputs for benchmarking using a Traditional Frontier (TF) according to Risk Averse (RA) preferences
    XOBS, YOBS = splitMomentsToXY(moments_df)
    gX, gY = RApreferences(XOBS, YOBS)
    return XOBS, YOBS, gX, gY


def TF_RL(moments_df: pd.DataFrame) -> Tuple[np.ndarray]:
    # Converts a dataframe of statistical moments to inputs and outputs for benchmarking using a Traditional Frontier (TF) according to Risk Loving (RL) preferences
    XOBS, YOBS = splitMomentsToXY(moments_df)
    gX, gY = RLpreferences(XOBS, YOBS)
    return XOBS, YOBS, gX, gY


def OF_RL(moments_df: pd.DataFrame) -> Tuple[np.ndarray]:
    # Converts a dataframe of statistical moments to outputs for benchmarking using an Output Frontier (OF) according to Risk Loving (RL) preferences
    XOBS, YOBS = OFsplitMomentsToXY(moments_df)
    gX, gY = RLpreferences(XOBS, YOBS)
    return XOBS, YOBS, gX, gY


def MainProg_ComEffiRank(r: pd.DataFrame, index_rebdate: pd.Timestamp, moment_generating_func: Callable[[pd.DataFrame], pd.DataFrame], getXYgXgY: Callable[[pd.DataFrame], Tuple[np.ndarray]] = TF_RA) -> pd.DataFrame:
    mom_1y = moment_generating_func(r.loc[(index_rebdate - pd.DateOffset(years=1) + pd.DateOffset(months=1)):index_rebdate,])
    mom_3y = moment_generating_func(r.loc[(index_rebdate - pd.DateOffset(years=3) + pd.DateOffset(months=1)):index_rebdate,])
    mom_5y = moment_generating_func(r.loc[(index_rebdate - pd.DateOffset(years=5) + pd.DateOffset(months=1)):index_rebdate,])

    eff_mom_1y = DEAFDH_Moments_Eff(mom_1y, getXYgXgY).rename(columns={'FDH': 'FDH_1y', 'DEA': 'DEA_1y'})
    eff_mom_3y = DEAFDH_Moments_Eff(mom_3y, getXYgXgY).rename(columns={'FDH': 'FDH_3y', 'DEA': 'DEA_3y'})
    eff_mom_5y = DEAFDH_Moments_Eff(mom_5y, getXYgXgY).rename(columns={'FDH': 'FDH_5y', 'DEA': 'DEA_5y'})

    eff = pd.concat([eff_mom_1y, eff_mom_3y, eff_mom_5y], axis=1)
    eff['DEA_MH'] = ((eff_mom_1y['DEA_1y']*0.95) + (eff_mom_3y['DEA_3y']*(0.95**3)) + (eff_mom_5y['DEA_5y']*(0.95**5)))/3
    eff['FDH_MH'] = ((eff_mom_1y['FDH_1y']*0.95) + (eff_mom_3y['FDH_3y']*(0.95**3)) + (eff_mom_5y['FDH_5y']*(0.95**5)))/3

    return eff


def DEAFDH_Moments_Eff(moments_df: pd.DataFrame, getXYgXgY: Callable[[pd.DataFrame], Tuple[np.ndarray]] = TF_RA) -> pd.DataFrame:
    XOBS, YOBS, gX, gY = getXYgXgY(moments_df)

    eff = pd.DataFrame(data=pd.NA, index=moments_df.index, columns=["FDH", "DEA"])
    eff.loc[:, "FDH"] = dirDistF(
        XOBS, YOBS, gX, gY, XREF=XOBS, YREF=YOBS, useConvex=False
    )
    eff.loc[:, "DEA"] = dirDistF(
        XOBS, YOBS, gX, gY, XREF=XOBS, YREF=YOBS, useConvex=True
    )
    #eff.loc[:, "DEA"] = eff.loc[:, "FDH"]
    return eff


def MVSK(r: pd.DataFrame) -> pd.DataFrame:
    # Implements: Multi_Horizon_MVSK.m
    # Moment computations (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html#scipy.stats.moment)
    MVSK = pd.DataFrame(data=pd.NA, index=r.columns, columns=["M", "V", "S", "K"])
    MVSK.loc[:, "M"] = np.mean(r.to_numpy(), axis=0)
    MVSK.loc[:, "V"] = moment(r, 2)
    MVSK.loc[:, "S"] = moment(r, 3)
    MVSK.loc[:, "K"] = moment(r, 4)
    return MVSK


def LMoments(r: pd.DataFrame) -> pd.DataFrame:
    # Compute L-moments up to order 4.
    # r is a (t x n) return matrix with t periods and n assets
    return pd.DataFrame(data=lmo.l_moment(r, range(1,5), axis=0).T, index=r.columns, columns=["LM1", "LM2", "LM3", "LM4"])


def TLMoments(r: pd.DataFrame, trim: Tuple[int]) -> pd.DataFrame:
    # Compute L-moments up to order 4.
    # r is a (t x n) return matrix with t periods and n assets
    return pd.DataFrame(data=lmo.l_moment(r, range(1,5), trim=trim, axis=0).T, index=r.columns, columns=["LM1", "LM2", "LM3", "LM4"])


def dirDistF(
    XOBS: np.ndarray,
    YOBS: np.ndarray,
    gX: np.ndarray,
    gY: np.ndarray,
    XREF: np.ndarray,
    YREF: np.ndarray,
    useConvex: bool,
) -> np.ndarray:
    
    effFDH = -np.inf * np.ones(XOBS.shape[0])
    for ind in range(XOBS.shape[0]):
        res = dirDistFDHVRS(XREF, YREF, XOBS[ind, :], YOBS[ind, :], gX[ind, :], gY[ind, :])
        effFDH[ind] = res["eff"]

    if useConvex:
        (effFDH_ind,) = np.nonzero(np.abs(effFDH) < 1e-6)
        eff = -np.inf * np.ones(XOBS.shape[0])
        for ind in range(XOBS.shape[0]):
            res = dirDistDEAVRS(XREF[effFDH_ind, :], YREF[effFDH_ind, :], XOBS[ind, :], YOBS[ind, :], gX[ind, :], gY[ind, :])
            eff[ind] = res["eff"]
    else:
        eff = effFDH

    return eff


class MVSKRating(bt.Algo):
    def __init__(self, moment_generating_func: Callable[[pd.DataFrame], pd.DataFrame] = MVSK, getXYgXgY: Callable[[pd.DataFrame], Tuple[np.ndarray]] = TF_RA, nselectassets: int = 30, useConvex: bool = False, max_window_years: int = 5, nr_moments: int = 4, returns_df: Optional[pd.DataFrame] = None):
        super().__init__()
        self.moment_func = moment_generating_func
        self.getXYgXgY = getXYgXgY
        self.nselectassets = nselectassets
        self.useConvex = useConvex
        self.max_window_years = max_window_years
        self.nr_moments = nr_moments
        self.returns_df = returns_df
    
    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now
        if self.returns_df is None:
            r = ffn.core.to_returns(target.universe.loc[:, selected]).dropna()
        else:
            # Dataframe of returns was also passed in
            r = self.returns_df.loc[:, selected]

        mom_1y = self.moment_func(r.loc[(t0 - pd.DateOffset(years=1) + pd.DateOffset(months=1)):t0,]).iloc[:, :self.nr_moments]
        XOBS, YOBS, gX, gY = self.getXYgXgY(mom_1y)
        eff_mom_1y = pd.Series(data=dirDistF(XOBS, YOBS, gX, gY, XREF=XOBS, YREF=YOBS, useConvex=self.useConvex), index=selected, name="1y")

        mom_3y = self.moment_func(r.loc[(t0 - pd.DateOffset(years=3) + pd.DateOffset(months=1)):t0,]).iloc[:, :self.nr_moments]
        XOBS, YOBS, gX, gY = self.getXYgXgY(mom_3y)
        eff_mom_3y = pd.Series(dirDistF(XOBS, YOBS, gX, gY, XREF=XOBS, YREF=YOBS, useConvex=self.useConvex), index=selected, name="3y")

        #mom_5y = self.moment_func(r.loc[(t0 - pd.DateOffset(years=5) + pd.DateOffset(months=1)):t0,]).iloc[:, :self.nr_moments]
        mom_5y = self.moment_func(r.loc[(t0 - pd.DateOffset(years=self.max_window_years) + pd.DateOffset(months=1)):t0,]).iloc[:, :self.nr_moments]
        XOBS, YOBS, gX, gY = self.getXYgXgY(mom_5y)
        eff_mom_5y = pd.Series(dirDistF(XOBS, YOBS, gX, gY, XREF=XOBS, YREF=YOBS, useConvex=self.useConvex), index=selected, name="5y")

        eff = pd.concat([eff_mom_1y, eff_mom_3y, eff_mom_5y], axis=1)
        #eff['MH'] = ((eff_mom_1y*0.95) + (eff_mom_3y*(0.95**3)) + (eff_mom_5y*(0.95**5)))/3
        eff['MH'] = ((eff_mom_1y*0.95) + (eff_mom_3y*(0.95**3)) + (eff_mom_5y*(0.95**self.max_window_years)))/3

        Rankres = eff['MH'].rank(method='min')

        PossibleSelectAsset = Rankres.index[Rankres <= self.nselectassets].tolist()
        if len(PossibleSelectAsset) <= self.nselectassets:
            SelectAsset_Equ = PossibleSelectAsset
            SelectAsset_More = []
            SelectAsset = SelectAsset_Equ
        else:
            SelectAsset_Equ = []
            SelectAsset_More = PossibleSelectAsset[0:(self.nselectassets-1)]
            # SelectAsset_More=PossibleSelectAsset(randperm(numel(PossibleSelectAsset),self.nselectassets));
            SelectAsset = SelectAsset_More
        
        target.temp["weights"] = {el: 1/self.nselectassets if el in SelectAsset else 0 for el in selected}
        
        return True