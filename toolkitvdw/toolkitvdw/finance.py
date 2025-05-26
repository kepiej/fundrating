# This library contains useful constructs related to mathematical finance.
# This library is based on the theory developed in the course text on
# mathematical finance by the author.
# (c) 2017 Ignace Van de Woestyne

import numpy as np
from scipy.optimize import minimize, brentq, linprog
from random import random, seed, shuffle
from datetime import *
from toolkitvdw.optimization import *


###################
# Data structures #
###################

class StatsMV:
    def __init__(self):
        self.M = None
        self.V = None


class StatsMVS:
    def __init__(self):
        self.M = None
        self.V = None
        self.S = None


class StatsMVSK:
    def __init__(self):
        self.M = None
        self.V = None
        self.S = None
        self.K = None


class CoMV:
    def __init__(self, m=None, v=None):
        self.M = m
        self.V = v

    def setMV(self, co):
        self.M = co[0]
        self.V = co[1]

    def setVM(self, co):
        self.V = co[0]
        self.M = co[1]

    def getMV(self):
        return [self.M, self.V]

    def getVM(self):
        return [self.V, self.M]


class CoMVS:
    def __init__(self, m=None, v=None, s=None):
        self.M = m
        self.V = v
        self.S = s

    def setMVS(self, co):
        self.M = co[0]
        self.V = co[1]
        self.S = co[2]

    def getMVS(self):
        return [self.M, self.V, self.S]


class CoMVSK:
    def __init__(self, m=None, v=None, s=None, k=None):
        self.M = m
        self.V = v
        self.S = s
        self.K = k

    def setMVSK(self, co):
        self.M = co[0]
        self.V = co[1]
        self.S = co[2]
        self.K = co[3]

    def getMVSK(self):
        return [self.M, self.V, self.S, self.K]


class OptEffMV:
    def __init__(self):
        self.eff = None
        self.success = None
        self.status = None
        self.xopt = None
        self.Mweak = None
        self.Vweak = None
        self.Mstrong = None
        self.Vstrong = None

    def getMVweak(self):
        return [self.Mweak, self.Vweak]

    def getMVstrong(self):
        return [self.Mstrong, self.Vstrong]


class OptEffMVS:
    def __init__(self):
        self.eff = None
        self.status = None
        self.success = None
        self.xopt = None
        self.Mweak = None
        self.Vweak = None
        self.Sweak = None
        self.Mstrong = None
        self.Vstrong = None
        self.Sstrong = None

    def getMVSweak(self):
        return [self.Mweak, self.Vweak, self.Sweak]

    def getMVSstrong(self):
        return [self.Mstrong, self.Vstrong, self.Sstrong]


class OptEffMVSK:
    def __init__(self):
        self.eff = None
        self.status = None
        self.success = None
        self.xopt = None
        self.Mweak = None
        self.Vweak = None
        self.Sweak = None
        self.Kweak = None
        self.Mstrong = None
        self.Vstrong = None
        self.Sstrong = None
        self.Kstrong = None

    def getMVSKweak(self):
        return [self.Mweak, self.Vweak, self.Sweak, self.Kweak]

    def getMVSKstrong(self):
        return [self.Mstrong, self.Vstrong, self.Sstrong, self.Kstrong]


class CashFlow:
    def __init__(self):
        self.cf = {}
        self.n = 0
        self.r = None

    def add(self, t, c):
        self.cf[t] = c
        self.n = len(self.cf.keys())

    def remove(self, t):
        self.cf.pop(t, None)
        self.n = len(self.cf.keys())

    def asList(self):
        return list(self.cf.items())

    def balancedAt(self, t, r, compType="effective"):
        if compType == "effective":
            def f(t):
                return 1 / (1 + r) ** t
        elif compType == "continuous":
            def f(t):
                return np.exp(-r * t)
        return -sum(self.cf[i] * f(i) for i in self.cf.keys()) / f(t)

    def irr(self, compType="effective"):
        def eq(r):
            if compType == "effective":
                def f(t):
                    return 1 / (1 + r) ** t
            elif compType == "continuous":
                def f(t):
                    return np.exp(-r * t)
            return sum(self.cf[i] * f(i) for i in self.cf.keys())

        try:
            res = brentq(eq, -0.9, 10)
            return res
        except:
            return np.nan


#############################
# Interest rate conversions #
#############################

def convertInterestRate(r, fromNrOfUnitPeriods, toNrOfUnitPeriods,
                        compType="effective"):
    def eq(rOut):
        if compType == "effective":
            return (1 + r) ** fromNrOfUnitPeriods - \
                   (1 + rOut) ** toNrOfUnitPeriods
        elif compType == "continuous":
            return  r * fromNrOfUnitPeriods - \
                    rOut * toNrOfUnitPeriods

    try:
        res = brentq(eq, -0.9, 10)
        return res
    except:
        return np.nan


def effectiveToContinousCompound(r):
    return np.log(1 + r)


def continuousCompoundToEffective(rcc):
    return np.exp(rcc) - 1


#####################
# Manipulating data #
#####################

# Check if date1 is situated before date2. Both dates must be arranged in
# the format determined by dateFormat (e.g., "%d/%m/%Y"). The result is
# either True or False. If both dates are identical, then the result is False.
def isBefore(date1, date2, dateFormat):
    date1 = datetime.strptime(date1, dateFormat)
    date2 = datetime.strptime(date2, dateFormat)
    if date1.year > date2.year:
        return False
    if date1.year < date2.year:
        return True
    if date1.year == date2.year:
        if date1.month > date2.month:
            return False
        if date1.month < date2.month:
            return True
        if date1.month == date2.month:
            if date1.day > date2.day:
                return False
            if date1.day < date2.day:
                return True
            if date1.day == date2.day:
                return False


# Provide a list with all positions in header whose dates are located
# between startDate and endDate. Both dates must be arranged in the format
# determined by dateFormat. Note that startDate and endDate are not
# included in the list.
def selectDateList(header, startDate, endDate, dateFormat):
    n = len(header)
    dateList = []
    for i in range(n):
        try:
            res1 = isBefore(startDate, header[i], dateFormat)
            res2 = isBefore(header[i], endDate, dateFormat)
            if res1 and res2:
                dateList.append(i)
        except:
            continue
    return dateList


# Return the position of headerItem in header assuming the header only
# contains dates. If headerItem is not present, then "NA" is returned.
def selectDateHeaderItem(header, headerItem, dateFormat):
    n = len(header)
    date1 = datetime.strptime(headerItem, dateFormat)
    for i in range(n):
        date2 = datetime.strptime(header[i], dateFormat)
        res1 = date2.year == date1.year
        res2 = date2.month == date1.month
        res3 = date2.day == date1.day
        if res1 and res2 and res3:
            return i
    return "NA"


# Return the position of headerItem in header. The header can contain
# anything. However, finding an item requires items to be identical.
# When header only contains dates, consider using selectDateHeaderItem
# instead which allows slight variations in the date (e.g., leading zeroes).
def selectHeaderItem(header, headerItem):
    n = len(header)
    for i in range(n):
        if header[i] == headerItem:
            return i
    return "NA"


# Convert day, month and year info to a string in the format provided by
# dateFormat.
def dayMonthYearToString(day, month, year, dateFormat):
    date1 = datetime(year, month, day)
    return date1.strftime(dateFormat)


# Return the positions of all days in the n year period in header prior
# to the date with index indexNr.
def selectNyearPeriod(header, indexNr, n, dateFormat):
    endDate = header[indexNr]
    date1 = datetime.strptime(endDate, dateFormat)
    d = date1.day
    m = date1.month
    y = date1.year - n
    startDate = dayMonthYearToString(d, m, y, dateFormat)
    return selectDateList(header, startDate, endDate, dateFormat)


# Return the positions of all days in the n month period in header prior
# to the date with index indexNr.
def selectNmonthPeriod(header, indexNr, n, dateFormat):
    endDate = header[indexNr]
    date1 = datetime.strptime(endDate, dateFormat)
    d = date1.day
    m = date1.month - n
    y = date1.year
    if m <= 0:
        m += 12
        y -= 1
    startDate = dayMonthYearToString(d, m, y, dateFormat)
    return selectDateList(header, startDate, endDate, dateFormat)


# Provide a list with all monthly positions in header whose dates are
# located between startDate and endDate. Both dates must be arranged in
# the format determined by dateFormat. The first date in header between
# startDate and endDate is always selected. Following dates are selected
# if the month differs from the previous month. Consequently, dates in
# the beginning of each month will be selected, except for possibly the
# first date (which is always selected). To realize a uniform spread,
# startDate should be situated at the end or beginning of a month (e.g.,
# last day of previous month or first day of new month).
def selectDateListMonthly(header, startDate, endDate, dateFormat):
    calendarList = selectDateList(header, startDate, endDate, dateFormat)
    n = len(calendarList)
    dateList = []
    monthNr = 0
    for i in range(n):
        indexNr = calendarList[i]
        date1 = datetime.strptime(header[indexNr], dateFormat)
        if date1.month != monthNr:
            monthNr = date1.month
            dateList.append(indexNr)
    return dateList


# Search for missing data present in dataMatrix. Missing data is supposed
# to be tagged with np.nan. The list of row numbers referring to these
# records is returned.
def findMissingData(dataMatrix):
    indexList = []
    n = len(dataMatrix)
    m = len(dataMatrix[0])
    for i in range(n):
        for j in range(m):
            if np.isnan(dataMatrix[i, j]):
                indexList.append(i)
                break
    return indexList


# Selects a sub (price) matrix of the (price) matrix p consisting of nObs
# observations directly preceding headerItem which should be present in
# header. If headerItem does not occur in header, then "NA" is returned.
# All dates must follow the format provided by dateFormat.
def selectMatrix(p, header, headerItem, nObs, dateFormat):
    endPos = selectDateHeaderItem(header, headerItem, dateFormat)
    if endPos == "NA":
        return "NA"
    else:
        endPos -= 1
        startPos = max(endPos - nObs + 1, 0)
        return np.array([p[i] for i in range(startPos, endPos + 1)])


# Selects a sub (price) matrix of the (price) matrix p consisting of all
# observations with index in headerList.
def extractMatrixFromHeaderList(p, headerList):
    return np.array([p[i] for i in headerList])


#########################
# Computing raw returns #
#########################

# Compute the matrix of raw returns from a given price matrix (without dates)
# expressed as effective interest rates. If a price is not available, np.nan is
# listed in the raw return matrix.
def effectiveRawReturnMatrix(priceMatrix):
    p = np.array(priceMatrix)
    nRow = len(p)
    nCol = len(p[0])
    res = np.zeros((nRow - 1, nCol))
    for i in range(1, nRow):
        for j in range(nCol):
            if np.isnan(p[i, j]) or np.isnan(p[i - 1, j]):
                res[i - 1, j] = np.nan
            else:
                res[i - 1, j] = (p[i, j] - p[i - 1, j]) / p[i - 1, j]
    return res


# Compute the matrix of raw returns from a given price matrix (without dates)
# expressed as continuous compound interest rates. If a price is not available,
# np.nan is listed in the raw return matrix.
def contCompoundRawReturnMatrix(priceMatrix):
    p = np.array(priceMatrix)
    nRow = len(p)
    nCol = len(p[0])
    res = np.zeros((nRow - 1, nCol))
    for i in range(1, nRow):
        for j in range(nCol):
            if np.isnan(p[i, j]) or np.isnan(p[i - 1, j]):
                res[i - 1, j] = np.nan
            else:
                res[i - 1, j] = np.log(p[i, j] / p[i - 1, j])
    return res


###########################
# Computing multi-moments #
###########################

# Compute the average return vector:
def computeM(r):
    return r.mean(axis=0)


# Compute the covariance matrix:
def computeV(r):
    r = np.array(r)
    avgRet = computeM(r)
    t = len(r)
    return np.einsum('ti,tj->ij', r - avgRet, r - avgRet) / t


# Compute the coskewness tensor:
def computeS(r):
    r = np.array(r)
    avgRet = computeM(r)
    t = len(r)
    return np.einsum('ti,tj,tk->ijk', r - avgRet, r - avgRet, r - avgRet) / t


# Compute the cokurtosis tensor:
def computeK(r):
    r = np.array(r)
    avgRet = computeM(r)
    t = len(r)
    return np.einsum('ti,tj,tk,tl->ijkl', r - avgRet, r - avgRet, r - avgRet, r - avgRet) / t


# Compute the average return of a portfolio x if the average return vector
# avgRet is already available. Note that avgRet can be computed using
# computeM, or can be extracted from the output of computeStatisticsMV or
# computeStatisticsMVS
def computeAverageReturn(x, avgRet):
    return np.dot(avgRet, x)


# Compute the variance of a portfolio x if the covariance matrix sigma
# is already available. Note that sigma can be computed using computeV,
# or can be extracted from the output of computeStatisticsMV or
# computeStatisticsMVS
def computeVariance(x, sigma):
    return np.dot(np.dot(sigma, x), x)


# Compute the skewness of a portfolio x if the coskewness tensor skew
# is already available. Note that skew can be computed using computeS,
# or can be extracted from the output of computeStatisticsMVS
def computeSkewness(x, skew):
    return np.dot(np.dot(np.dot(skew, x), x), x)


# Compute the kurtosis of a portfolio x if the cokurtosis tensor kurt
# is already available. Note that kurt can be computed using computeK
def computeKurtosis(x, kurt):
    return np.dot(np.dot(np.dot(np.dot(kurt, x), x), x), x)


# Compute the average return of a portfolio x taken from the financial
# universe determined by the raw return matrix r.
def portfolioAverageReturn(x, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    return computeM(rstar)[0]


# Compute the variance of a portfolio x taken from the financial
# universe determined by the raw return matrix r.
def portfolioVariance(x, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    return computeV(rstar)[0][0]


# Compute the skewness of a portfolio x taken from the financial
# universe determined by the raw return matrix r.
def portfolioSkewness(x, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    return computeS(rstar)[0][0][0]


# Compute the kurtosis of a portfolio x taken from the financial
# universe determined by the raw return matrix r.
def portfolioKurtosis(x, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    return computeK(rstar)[0][0][0][0]


# Compute the moment of order p of a portfolio x taken from the
# financial universe determined by the raw return matrix r.
def portfolioMultimoment(x, p, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    rmean = np.mean(rstar)
    return np.mean((rstar - rmean) ** p)


#################
# MV statistics #
#################

# Compute the average return vector and covariance matrix
# of the assets given in the return matrix r (columns contain the returns of
# the individual assets).
def computeStatisticsMV(r):
    sol = StatsMV()
    sol.M = computeM(r)
    sol.V = computeV(r)
    return sol


# Compute the average return and variance of one asset from the given return
# matrix r (columns contain the returns of the individual assets).
#  input: r = return matrix, i = number of the asset under observation (starts from 0)
# output: data structure containing average return and variance
def computeStatisticsMVoneAsset(i, r):
    sol = StatsMV()
    r = np.array(r)
    asset = r[:, i].reshape(-1, 1)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    return sol


# Compute the average return and variance of a portfolio determined by its weight
# vector x with respect to the universe of returns r (column contain the returns
# of the individual assets in the universe).
def computeStatisticsMVonePortfolio(x, r):
    sol = StatsMV()
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    asset = np.dot(r, x)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    return sol


##################
# MVS statistics #
##################

# Compute the average return vector, covariance matrix and coskewness tensor
# of the assets given in the return matrix r (columns contain the returns of
# the individual assets).
def computeStatisticsMVS(r):
    sol = StatsMVS()
    sol.M = computeM(r)
    sol.V = computeV(r)
    sol.S = computeS(r)
    return sol


# Compute the average return, variance and skewness of one asset from the given
# return matrix r (columns contain the returns of the individual assets).
#  input: r = return matrix, i = number of the asset under observation (starts from 0)
# output: data structure containing average return, variance and skewness
def computeStatisticsMVSoneAsset(i, r):
    sol = StatsMVS()
    r = np.array(r)
    asset = r[:, i].reshape(-1, 1)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    sol.S = computeS(asset)[0][0][0]
    return sol


# Compute the average return, variance and skewness of a portfolio determined
# by its weight vector x with respect to the universe of returns r (column
# contain the returns of the individual assets in the universe).
def computeStatisticsMVSonePortfolio(x, r):
    sol = StatsMVS()
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    asset = np.dot(r, x)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    sol.S = computeS(asset)[0][0][0]
    return sol


###################
# MVSK statistics #
###################

# Compute the average return vector, covariance matrix, coskewness tensor
# and cokurtosis tensor of the assets given in the return matrix r
# (columns contain the returns of the individual assets).
def computeStatisticsMVSK(r):
    sol = StatsMVSK()
    sol.M = computeM(r)
    sol.V = computeV(r)
    sol.S = computeS(r)
    sol.K = computeK(r)
    return sol


# Compute the average return, variance, skewness and kurtosis of one asset
# from the given return matrix r (columns contain the returns of the
# individual assets).
# input: r = return matrix, i = number of the asset under observation
# (starts from 0)
# output: data structure containing average return, variance, skewness
# and kurtosis
def computeStatisticsMVSKoneAsset(i, r):
    sol = StatsMVSK()
    r = np.array(r)
    asset = r[:, i].reshape(-1, 1)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    sol.S = computeS(asset)[0][0][0]
    sol.K = computeK(asset)[0][0][0][0]
    return sol


# Compute the average return, variance, skewness and kurtosis of a portfolio
# determined by its weight vector x with respect to the universe of returns
# r (columns contain the returns of the individual assets in the universe).
def computeStatisticsMVSKonePortfolio(x, r):
    sol = StatsMVS()
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    asset = np.dot(r, x)
    sol.M = computeM(asset)[0]
    sol.V = computeV(asset)[0][0]
    sol.S = computeS(asset)[0][0][0]
    sol.K = computeK(asset)[0][0][0][0]
    return sol


##########################
# Alternative statistics #
##########################

# Compute the semivariance of a portfolio:
def portfolioSemivariance(x, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    rmean = np.mean(rstar)
    rstar = rstar.reshape(1, -1)[0]
    return np.mean([max(rmean - element, 0) ** 2 for element in rstar])


# Compute the semicovariance of two assets:
def semicovariance(j, k, r):
    r = np.array(r)
    t = len(r)
    avgRet = computeM(r)
    return np.mean([max(avgRet[j] - r[i, j], 0) * max(avgRet[k] - r[i, k], 0) for i in range(t)])


# Compute the semicovariance matrix:
def semicovarianceMatrix(r):
    n = len(r[0])
    mat = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            mat[j, k] = semicovariance(j, k, r)
    return mat


# Compute the lower partial moment of a portfolio:
def portfolioLPM(x, p, z0, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    rstar = rstar.reshape(1, -1)[0]
    return np.mean([max(z0 - element, 0) ** p for element in rstar])


# Compute the upper partial moment of a portfolio:
def portfolioUPM(x, p, z0, r):
    r = np.array(r)
    x = np.array(x).reshape(-1, 1)
    rstar = np.dot(r, x)
    rstar = rstar.reshape(1, -1)[0]
    return np.mean([max(element - z0, 0) ** p for element in rstar])


#########################
# Shortage functions MV #
#########################

def shMVgeneralDirectionOld(obs, g, stats, nRestart=1, equality=False,
                         shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V

    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMV()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.success = resMin.success
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
    return sol

def shMVgeneralDirection(obs, g, stats, nRestart=0, equality=False,
                         shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V

    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)

    def fvar(x):
        return obs.V + x[n] * g.V - computeVariance(x[:n], sigma)

    if equality:
        consEq = [fsum, fret, fvar]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    start = [0.1 for j in range(n)] + [1]
    start = np.array(start)
    res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                   gtcon=consGt, minimization=False)
    if res.success:
        if res.fun > optMax:
            optMax = res.fun
            resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = np.sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMV()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.success = resMax.success
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
    return sol


def shMV(obs, stats, nRestart=0, equality=False, shortselling=False, deltaPos=True):
    g = CoMV(abs(obs.M), -abs(obs.V))
    return shMVgeneralDirection(obs, g, stats, nRestart, equality, shortselling, deltaPos)


def shMVgeneralDirectionWithRf(obs, g, stats, rf, nRestart=1, equality=False,
                               shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V

    def fobj(x):
        return x[n + 1]

    def fsum(x):
        return sum(x[:n + 1]) - 1

    def fret(x):
        return -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]

    def fvar(x):
        return obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)

    if equality:
        consEq = [fsum, fret, fvar]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n + 1)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMV()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.success = resMax.success
        sol.xopt = resMax.x[:n + 1]
        sol.Mweak = obs.M + resMax.x[n + 1] * g.M
        sol.Vweak = obs.V + resMax.x[n + 1] * g.V
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet) + rf * resMax.x[n]
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
    return sol


def shMVgeneralDirectionWithRfOld(obs, g, stats, rf, nRestart=1, equality=False,
                                  shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V

    def func(x):
        return -x[n + 1]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n + 1)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMV()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.success = resMin.success
        sol.xopt = resMin.x[:n + 1]
        sol.Mweak = obs.M + resMin.x[n + 1] * g.M
        sol.Vweak = obs.V + resMin.x[n + 1] * g.V
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet) + rf * resMin.x[n]
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
    return sol


def shMVgeneralDirectionAndMoment(obs, g, n, momentM, momentV, nRestart=1,
                                  equality=False, shortselling=False, deltaPos=True):

    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + momentM(x[:n])

    def fvar(x):
        return obs.V + x[n] * g.V - momentV(x[:n])

    if equality:
        consEq = [fsum, fret, fvar]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMV()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.success = resMax.success
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Mstrong = momentM(resMax.x[:n])
        sol.Vstrong = momentV(resMax.x[:n])
    return sol


def shMVgeneralDirectionAndMomentOld(obs, g, n, momentM, momentV, nRestart=1,
                                     equality=False, shortselling=False, deltaPos=True):
    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMV()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.success = resMin.success
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Mstrong = momentM(resMin.x[:n])
        sol.Vstrong = momentV(resMin.x[:n])
    return sol


##########################
# Shortage functions MVS #
##########################

def shMVSgeneralDirection(obs, g, stats, nRestart=0, equality=False,
                          shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S

    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)

    def fvar(x):
        return obs.V + x[n] * g.V - computeVariance(x[:n], sigma)

    def fskew(x):
        return -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)

    if equality:
        consEq = [fsum, fret, fvar, fskew]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVS()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.success = resMax.success
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Sweak = obs.S + resMax.x[n] * g.S
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMax.x[:n], skew)
    return sol


def shMVSgeneralDirectionOld(obs, g, stats, nRestart=0, equality=False,
                             shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S

    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVS()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.success = resMin.success
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Sweak = obs.S + resMin.x[n] * g.S
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMin.x[:n], skew)
    return sol


def shMVS(obs, stats, nRestart=0, equality=False, shortselling=False, deltaPos=True):
    g = CoMVS(abs(obs.M), -abs(obs.V), abs(obs.S))
    return shMVSgeneralDirection(obs, g, stats, nRestart, equality, shortselling, deltaPos)


def shMVSgeneralDirectionWithRf(obs, g, stats, rf, nRestart=0, equality=False,
                                shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S

    def fobj(x):
        return x[n + 1]

    def fsum(x):
        return sum(x[:n + 1]) - 1

    def fret(x):
        return -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]

    def fvar(x):
        return obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)

    def fskew(x):
        return -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)

    if equality:
        consEq = [fsum, fret, fvar, fskew]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n + 1)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n + 1):
        start = [0 for j in range(n + 2)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVS()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.xopt = resMax.x[:n + 1]
        sol.Mweak = obs.M + resMax.x[n + 1] * g.M
        sol.Vweak = obs.V + resMax.x[n + 1] * g.V
        sol.Sweak = obs.S + resMax.x[n + 1] * g.S
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet) + rf * resMax.x[n]
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMax.x[:n], skew)
    return sol


def shMVSgeneralDirectionWithRfOld(obs, g, stats, rf, nRestart=0, equality=False,
                                   shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S

    def func(x):
        return -x[n + 1]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n + 1):
        start = [0 for j in range(n + 2)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n + 1)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n + 1)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVS()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.xopt = resMin.x[:n + 1]
        sol.Mweak = obs.M + resMin.x[n + 1] * g.M
        sol.Vweak = obs.V + resMin.x[n + 1] * g.V
        sol.Sweak = obs.S + resMin.x[n + 1] * g.S
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet) + rf * resMin.x[n]
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMin.x[:n], skew)
    return sol


def shMVSgeneralDirectionAndMoment(obs, g, n, momentM, momentV, momentS, nRestart=0,
                                   equality=False, shortselling=False, deltaPos=True):
    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + momentM(x[:n])

    def fvar(x):
        return obs.V + x[n] * g.V - momentV(x[:n])

    def fskew(x):
        return -obs.S - x[n] * g.S + momentS(x[:n])

    if equality:
        consEq = [fsum, fret, fvar, fskew]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVS()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Sweak = obs.S + resMax.x[n] * g.S
        sol.Mstrong = momentM(resMax.x[:n])
        sol.Vstrong = momentV(resMax.x[:n])
        sol.Sstrong = momentS(resMax.x[:n])
    return sol


def shMVSgeneralDirectionAndMomentOld(obs, g, n, momentM, momentV, momentS, nRestart=0,
                                   equality=False, shortselling=False, deltaPos=True):
    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n] * g.S + momentS(x[:n])})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n] * g.S + momentS(x[:n])})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [deltaPos])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVS()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Sweak = obs.S + resMin.x[n] * g.S
        sol.Mstrong = momentM(resMin.x[:n])
        sol.Vstrong = momentV(resMin.x[:n])
        sol.Sstrong = momentS(resMin.x[:n])
    return sol


###########################
# Shortage functions MVSK #
###########################

def shMVSKgeneralDirection(obs, g, stats, nRestart=0, equality=False,
                           shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S
    kurt = stats.K

    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)

    def fvar(x):
        return obs.V + x[n] * g.V - computeVariance(x[:n], sigma)

    def fskew(x):
        return -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)

    def fkurt(x):
        return obs.K + x[n] * g.K - computeKurtosis(x[:n], kurt)

    if equality:
        consEq = [fsum, fret, fvar, fskew, fkurt]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew, fkurt]

    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVSK()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.success = resMax.success
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Sweak = obs.S + resMax.x[n] * g.S
        sol.Kweak = obs.K + resMax.x[n] * g.K
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMax.x[:n], skew)
        sol.Kstrong = computeKurtosis(resMax.x[:n], kurt)
    return sol


def shMVSKgeneralDirectionOld(obs, g, stats, nRestart=0, equality=False,
                              shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S
    kurt = stats.K

    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)},
                {'type': 'ineq',
                 'fun': lambda x: obs.K + x[n] * g.K - computeKurtosis(x[:n], kurt)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + computeAverageReturn(x[:n], avgRet)},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - computeVariance(x[:n], sigma)},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n] * g.S + computeSkewness(x[:n], skew)},
              {'type': 'eq',
               'fun': lambda x: obs.K + x[n] * g.K - computeKurtosis(x[:n], kurt)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVSK()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.success = resMin.success
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Sweak = obs.S + resMin.x[n] * g.S
        sol.Kweak = obs.K + resMin.x[n] * g.K
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet)
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMin.x[:n], skew)
        sol.Kstrong = computeKurtosis(resMin.x[:n], kurt)
    return sol


def shMVSK(obs, stats, nRestart=0, equality=False, shortselling=False, deltaPos=True):
    g = CoMVSK(abs(obs.M), -abs(obs.V), abs(obs.S), -abs(obs.K))
    return shMVSKgeneralDirection(obs, g, stats, nRestart, equality, shortselling, deltaPos)


def shMVSKgeneralDirectionWithRf(obs, g, stats, rf, nRestart=0, equality=False,
                                 shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S
    kurt = stats.K

    def fobj(x):
        return x[n + 1]

    def fsum(x):
        return sum(x[:n + 1]) - 1

    def fret(x):
        return -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]

    def fvar(x):
        return obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)

    def fskew(x):
        return -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)

    def fkurt(x):
        return obs.K + x[n + 1] * g.K - computeKurtosis(x[:n], kurt)

    if equality:
        consEq = [fsum, fret, fvar, fskew, fkurt]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew, fkurt]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n + 1)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n + 1):
        start = [0 for j in range(n + 2)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVSK()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.xopt = resMax.x[:n + 1]
        sol.Mweak = obs.M + resMax.x[n + 1] * g.M
        sol.Vweak = obs.V + resMax.x[n + 1] * g.V
        sol.Sweak = obs.S + resMax.x[n + 1] * g.S
        sol.Kweak = obs.K + resMax.x[n + 1] * g.K
        sol.Mstrong = computeAverageReturn(resMax.x[:n], avgRet) + rf * resMax.x[n]
        sol.Vstrong = computeVariance(resMax.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMax.x[:n], skew)
        sol.Kstrong = computeKurtosis(resMax.x[:n], kurt)
    return sol


def shMVSKgeneralDirectionWithRfOld(obs, g, stats, rf, nRestart=0, equality=False,
                                    shortselling=False, deltaPos=True):
    n = len(stats.M)
    avgRet = stats.M
    sigma = stats.V
    skew = stats.S
    kurt = stats.K

    def func(x):
        return -x[n + 1]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)},
                {'type': 'ineq',
                 'fun': lambda x: obs.K + x[n + 1] * g.K - computeKurtosis(x[:n], kurt)})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n + 1]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n + 1] * g.M + computeAverageReturn(x[:n], avgRet) + rf * x[n]},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n + 1] * g.V - computeVariance(x[:n], sigma)},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n + 1] * g.S + computeSkewness(x[:n], skew)},
              {'type': 'eq',
               'fun': lambda x: obs.K + x[n + 1] * g.K - computeKurtosis(x[:n], kurt)})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n + 1):
        start = [0 for j in range(n + 2)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n + 1)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n + 1)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n + 1)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVSK()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.xopt = resMin.x[:n + 1]
        sol.Mweak = obs.M + resMin.x[n + 1] * g.M
        sol.Vweak = obs.V + resMin.x[n + 1] * g.V
        sol.Sweak = obs.S + resMin.x[n + 1] * g.S
        sol.Kweak = obs.K + resMin.x[n + 1] * g.K
        sol.Mstrong = computeAverageReturn(resMin.x[:n], avgRet) + rf * resMin.x[n]
        sol.Vstrong = computeVariance(resMin.x[:n], sigma)
        sol.Sstrong = computeSkewness(resMin.x[:n], skew)
        sol.Kstrong = computeKurtosis(resMin.x[:n], kurt)
    return sol


def shMVSKgeneralDirectionAndMoment(obs, g, n, momentM, momentV, momentS, momentK, nRestart=0,
                                    equality=False, shortselling=False, deltaPos=True):
    def fobj(x):
        return x[n]

    def fsum(x):
        return sum(x[:n]) - 1

    def fret(x):
        return -obs.M - x[n] * g.M + momentM(x[:n])

    def fvar(x):
        return obs.V + x[n] * g.V - momentV(x[:n])

    def fskew(x):
        return -obs.S - x[n] * g.S + momentS(x[:n])

    def fkurt(x):
        return obs.K + x[n] * g.K - momentK(x[:n])

    if equality:
        consEq = [fsum, fret, fvar, fskew, fkurt]
        consGt = []
    else:
        consEq = [fsum]
        consGt = [fret, fvar, fskew, fkurt]
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    lstBounds = [bnd for i in range(n)] + [bndDelta]
    optMax = -1E06
    resMax = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = solveNLP(fobj, start, bounds=lstBounds, eqcon=consEq,
                       gtcon=consGt, minimization=False)
        if res.success:
            if res.fun > optMax:
                optMax = res.fun
                resMax = res
    sol = OptEffMVSK()
    if resMax is not None:
        sol.eff = resMax.fun
        sol.status = resMax.status
        sol.xopt = resMax.x[:n]
        sol.Mweak = obs.M + resMax.x[n] * g.M
        sol.Vweak = obs.V + resMax.x[n] * g.V
        sol.Sweak = obs.S + resMax.x[n] * g.S
        sol.Kweak = obs.K + resMax.x[n] * g.K
        sol.Mstrong = momentM(resMax.x[:n])
        sol.Vstrong = momentV(resMax.x[:n])
        sol.Sstrong = momentS(resMax.x[:n])
        sol.Kstrong = momentK(resMax.x[:n])
    return sol


def shMVSKgeneralDirectionAndMomentOld(obs, g, n, momentM, momentV, momentS, momentK, nRestart=0,
                                       equality=False, shortselling=False, deltaPos=True):
    def func(x):
        return -x[n]

    consIneq = ({'type': 'eq',
                 'fun': lambda x: sum(x[:n]) - 1},  # = 0
                {'type': 'ineq',
                 'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: -obs.S - x[n] * g.S + momentS(x[:n])},
                {'type': 'ineq',
                 'fun': lambda x: obs.K + x[n] * g.K - momentK(x[:n])})  # >= 0

    consEq = ({'type': 'eq',
               'fun': lambda x: sum(x[:n]) - 1},  # = 0
              {'type': 'eq',
               'fun': lambda x: -obs.M - x[n] * g.M + momentM(x[:n])},
              {'type': 'eq',
               'fun': lambda x: obs.V + x[n] * g.V - momentV(x[:n])},
              {'type': 'eq',
               'fun': lambda x: -obs.S - x[n] * g.S + momentS(x[:n])},
              {'type': 'eq',
               'fun': lambda x: obs.K + x[n] * g.K - momentK(x[:n])})

    if equality:
        cons = consEq
    else:
        cons = consIneq
    if shortselling:
        bnd = (-1E06, 1E06)
    else:
        bnd = (-1E-06, 1)
    if deltaPos:
        bndDelta = (-1E-06, 1E06)
    else:
        bndDelta = (-1E06, 1E06)
    optMin = 1E06
    resMin = None
    for i in range(n):
        start = [0 for j in range(n + 1)]
        start[i] = 1
        start = np.array(start)
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [bndDelta])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    for i in range(nRestart):
        start = [random() for j in range(n)]
        tot = sum(start)
        start = np.array([st / tot for st in start] + [random()])
        res = minimize(func, start, constraints=cons, method='SLSQP', options={'disp': False},
                       bounds=[bnd for i in range(n)] + [deltaPos])
        if res.success:
            if res.fun < optMin:
                optMin = res.fun
                resMin = res
    sol = OptEffMVSK()
    if resMin is not None:
        sol.eff = -resMin.fun
        sol.status = resMin.status
        sol.xopt = resMin.x[:n]
        sol.Mweak = obs.M + resMin.x[n] * g.M
        sol.Vweak = obs.V + resMin.x[n] * g.V
        sol.Sweak = obs.S + resMin.x[n] * g.S
        sol.Kweak = obs.K + resMin.x[n] * g.K
        sol.Mstrong = momentM(resMin.x[:n])
        sol.Vstrong = momentV(resMin.x[:n])
        sol.Sstrong = momentS(resMin.x[:n])
        sol.Kstrong = momentK(resMin.x[:n])
    return sol


###############
# Backtesting #
###############

# Fill in missing prices (i.e. np.nan) in the price matrix (dates in first column and no codes)
# by assuming the price remained unchanged compared to the previous period. Note
# that this procedure only succeeds in filling in all missing prices if there is at least one
# price available prior to the time these occur.
def completePriceMatrix(priceMatrix):
    priceMatrix = np.array(priceMatrix)
    n = len(priceMatrix)
    m = len(priceMatrix[0])
    for i in range(1, n):
        for j in range(m):
            if np.isnan(priceMatrix[i, j]):
                priceMatrix[i, j] = priceMatrix[i - 1, j]
    return priceMatrix


# Determine if rebalancing has occurred by examining changes in weightMatrix.
# If there is no change in the weights compared with the previous period, then
# "NR" (no rebalancing) is assigned, otherwise "R" (rebalancing) is assigned.
# A list containing the consecutive assignments is returned.
def determineRebalancingStatus(weightMatrix):
    n = len(weightMatrix)
    m = len(weightMatrix[0])
    res = ["R"]
    for i in range(1, n):
        diff = sum(abs(weightMatrix[i, j] - weightMatrix[i - 1, j]) for j in range(m))
        if diff < 0.000001:
            res.append("NR")
        else:
            res.append("R")
    return res


# Returns an adjusted weight matrix in which the weights are replaced by the
# code np.nan if no rebalancing has been detected. This way, the weight matrix
# is made compatible with the backtesting procedure computeEvolutionCapital.
def adjustWeightMatrix(weightMatrix):
    rebalancingStatus = determineRebalancingStatus(weightMatrix)
    n = len(weightMatrix)
    m = len(weightMatrix[0])
    res = np.zeros((n, m))
    for i in range(n):
        if rebalancingStatus[i] == "NR":
            for j in range(m):
                res[i, j] = np.nan
        else:
            for j in range(m):
                res[i, j] = weightMatrix[i, j]
    return res


# Computes the evolution of a portfolio's capital.
# - priceMatrix contains the asset prices for all periods in the calendar
# - weightMatrix contains the weights of the assets in the given portfolio
# for all periods in the calendar. If no rebalancing is required for a
# certain period in the calendar, then np.nan should be present in
# weightMatrix for all assets for that particular period. Note that np.nan
# is not allowed in the first date of the calendar.
# - kInitial equals the initial capital
# The procedure returns the following:
# - The evolution of the capital for all periods in the calendar
# - The "number of assets" for each period of the calendar computed from the
# portfolio weights.
def computeEvolutionCapital(priceMatrix, weightMatrix, kInitial):
    nFunds = len(priceMatrix[0])
    nCal = len(priceMatrix)
    n = np.zeros((nCal, nFunds))
    k = [kInitial]
    for j in range(nFunds):
        n[0, j] = weightMatrix[0, j] / priceMatrix[0, j] * k[0]
    for i in range(1, nCal):
        tot = sum(n[i - 1, j] * priceMatrix[i, j] for j in range(nFunds))
        k.append(tot)
        for j in range(nFunds):
            if np.isnan(weightMatrix[i, j]):
                n[i, j] = n[i - 1, j]
            else:
                n[i, j] = weightMatrix[i, j] / priceMatrix[i, j] * k[i]
    return k, n


# Compute the exit costs when selling units of assets. This procedure requires
# the price matrix priceMatrix, the matrix n containing the number of units of
# assets over all rebalancing moments, and the exit fee matrix omega containing
# the percentages to be applied on the volume sold for each asset at each
# rebalancing moment. Selling is assumed if the number of units of assets
# decreases compared to the previous period. A list containing the exit costs
# is returned.
def computeExitCosts(priceMatrix, n, omega):
    nFunds = len(priceMatrix[0])
    nCal = len(priceMatrix)
    cost = []
    cFund = np.zeros((nCal, nFunds))
    for i in range(1, nCal):
        for j in range(nFunds):
            cFund[i, j] = priceMatrix[i, j] * max(n[i - 1, j] - n[i, j], 0) * omega[i, j]
    for i in range(nCal):
        tot = sum(cFund[i, j] for j in range(nFunds))
        cost.append(tot)
    return cost


# Compute the exit costs when terminating the investment at a rebalancing
# moment. This procedure requires the price matrix priceMatrix, the matrix n
# containing the number of units of assets over all rebalancing moments, and
# the exit fee matrix omega containing the percentages to be applied on the
# volume sold for each asset at each rebalancing moment. A list containing the
# exit costs on terminate is returned.
def computeExitCostsOnTerminate(priceMatrix, n, omega):
    nFunds = len(priceMatrix[0])
    nCal = len(priceMatrix)
    cost = []
    cFund = np.zeros((nCal, nFunds))
    for i in range(1, nCal):
        for j in range(nFunds):
            cFund[i, j] = priceMatrix[i, j] * n[i - 1, j] * omega[i, j]
    for i in range(nCal):
        tot = sum(cFund[i, j] for j in range(nFunds))
        cost.append(tot)
    return cost


# Compute the entry costs when buying units of assets. This procedure
# requires the price matrix priceMatrix, the matrix n containing the number of
# units of assets over all rebalancing moments, and the entry fee matrix alpha
# containing the percentages to be applied on the volume bought for each
# asset at each rebalancing moment. Buying is assumed if the number of
# units of assets increases compared to the previous period. A list
# containing the entry costs is returned.
def computeEntryCosts(priceMatrix, n, alpha):
    nFunds = len(priceMatrix[0])
    nCal = len(priceMatrix)
    cost = []
    cFund = np.zeros((nCal, nFunds))
    for j in range(nFunds):
        cFund[0, j] = priceMatrix[0, j] * n[0, j] * alpha[0, j]
    for i in range(1, nCal):
        for j in range(nFunds):
            cFund[i, j] = priceMatrix[i, j] * max(n[i, j] - n[i - 1, j], 0) * alpha[i, j]
    for i in range(nCal):
        tot = sum(cFund[i, j] for j in range(nFunds))
        cost.append(tot)
    return cost


# Compute the accumulated total costs for each period in the calendar
# from the lists cAlpha and cOmega containing the entry and exit costs,
# respectively. The list returned can be considered as the evolution of
# the total costs over time.
# Note that this procedure does not consider an interest effect since it
# adds costs made at different moments in time. Therefore, this procedure
# should only be used for estimations. Consider using computeIRR instead
# to have a correct way of dealing with costs located over time.
def computeTotalCostCumul(cAlpha, cOmega):
    n = len(cAlpha)
    cCumul = [cAlpha[0] + cOmega[0]]
    for i in range(1, n):
        cCumul.append(cCumul[i - 1] + cAlpha[i] + cOmega[i])
    return cCumul


# Compute the internal rate of return (IRR) of the investment for each
# period in the calendar assuming that the investment is terminated
# (and therefore sold) at that period. This procedure requires the list
# containing the evolution of capital (evolutionC), the lists with total
# entry costs (entryCost), total exit costs (exitCost) and total exit
# costs on terminate (exitCostOnTerminate), and the rebalancing frequency
# in one year (nYear) (e.g., nYear = 12 in case of monthly rebalancing).
# The procedure returns a list with IRRs computed on a yearly basis.
#
# Note that the exit costs at that period are adjusted by this function
# since it does not make sense to rebalance and sell at the same time.
def computeIRR(evolutionC, entryCost, exitCost, exitCostOnTerminate, nYear):
    nCal = len(evolutionC)
    irr = [0]
    for i in range(1, nCal):
        def f(r):
            r1 = 1 / (1 + r)
            equ = -evolutionC[0] - sum(entryCost[j] * r1 ** j for j in range(i)) - \
                  sum(exitCost[j] * r1 ** j for j in range(i)) - \
                  exitCostOnTerminate[i] * r1 ** i + evolutionC[i] * r1 ** i
            return equ

        rm = brentq(f, -0.9, 10)
        irr.append((1 + rm) ** nYear - 1)
    return irr


##################################
# Directional distance functions #
##################################

def dirDistDEAVRSold(x, y, inputDMU, outputDMU, gInput, gOutput):
    gInput = -gInput  # Assume that the original gInput is negative
    n = len(x)
    p = len(x[0])
    q = len(y[0])
    rangeInput = [i for i in range(p)]
    rangeOutput = [i for i in range(q)]
    rangeDMU = [i for i in range(n)]
    listConstr = []
    for i in rangeInput:
        el = {}
        el["type"] = "ineq"

        def make_func(i):
            def f(z):
                res = inputDMU[i] - z[n] * gInput[i] - \
                      np.sum(z[j] * x[j][i] for j in rangeDMU)
                return res

            return f

        el["fun"] = make_func(i)
        listConstr += [el]
    for i in rangeOutput:
        el = {}
        el["type"] = "ineq"

        def make_func(i):
            def f(z):
                res = -outputDMU[i] - z[n] * gOutput[i] + \
                      np.sum(z[j] * y[j][i] for j in rangeDMU)
                return res

            return f

        el["fun"] = make_func(i)
        listConstr += [el]
    el = {}
    el["type"] = "eq"
    el["fun"] = lambda z: np.sum(z[j] for j in rangeDMU) - 1
    listConstr += [el]
    obj = lambda z: -z[n]
    listBounds = [(0, None) for i in rangeDMU] + [(None, None)]
    initPt = [0 for i in rangeDMU] + [1]
    res = minimize(obj, initPt, bounds=listBounds,
                   constraints=listConstr)
    res.x = res.x[:n]
    res.fun = -res.fun
    resDict = {}
    resDict["eff"] = res.fun
    resDict["status"] = res.status
    resDict["zOpt"] = res.x
    resDict["iOpt"] = None
    return resDict

# Assume gInput <= 0
def dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput):
    n = len(x)
    p = len(x[0])
    q = len(y[0])
    rangeDMU = [i for i in range(n)]
    c = [0 for i in rangeDMU]
    c.append(-1)
    aTop = np.hstack((np.transpose(x), -np.reshape(gInput, (-1, 1))))
    aBottom = np.hstack((-np.transpose(y), np.reshape(gOutput, (-1, 1))))
    aIneq = np.vstack((aTop, aBottom))
    bIneq = np.vstack((np.reshape(inputDMU, (-1, 1)),
                       -np.reshape(outputDMU, (-1, 1))))
    aEq = np.hstack((np.ones((1, n)), [[0]]))
    bEq = np.array([[1]])
    bnds = [(1E-5, 1) for i in rangeDMU]
    bnds.append((1E-5, 1000))
    res = linprog(c, A_ub=aIneq, b_ub=bIneq, A_eq=aEq, b_eq=bEq,
                  bounds=bnds)
    res.x = res.x[:n]
    res.fun = -res.fun
    resDict = {}
    resDict["eff"] = res.fun
    resDict["status"] = res.status
    resDict["zOpt"] = res.x
    resDict["iOpt"] = None
    return resDict


def dirDistFDHVRS(x, y, inputDMU, outputDMU, gInput, gOutput):
    gInput = -np.array(gInput)  # Assume that the original gInput is negative
    n = len(x)
    p = len(x[0])
    q = len(y[0])
    res = {}
    listgInputPos = []
    listgInputZero = []
    for j in range(p):
        if gInput[j] > 0:
            listgInputPos.append(j)
        if gInput[j] == 0:
            listgInputZero.append(j)
    listgOutputPos = []
    listgOutputZero = []
    for j in range(q):
        if gOutput[j] > 0:
            listgOutputPos.append(j)
        if gOutput[j] == 0:
            listgOutputZero.append(j)
    if len(listgInputZero) + len(listgOutputZero) == 0:
        domObs = [k for k in range(n)]
    else:
        domObs = []
        for i in range(n):
            dominate = True
            for j in listgOutputZero:
                if y[i, j] < outputDMU[j]:
                    dominate = False
                    break
            for j in listgInputZero:
                if x[i, j] > inputDMU[j]:
                    dominate = False
                    break
            if dominate:
                domObs.append(i)
    if domObs == []:
        res["status"] = 1
        res["eff"] = None
        res["zOpt"] = None
        res["iOpt"] = None
        return res
    lst = [min([(y[i, j] - outputDMU[j]) / gOutput[j] for j in listgOutputPos] + \
               [(inputDMU[j] - x[i, j]) / gInput[j] for j in listgInputPos]) for i in domObs]
    res["eff"] = max(lst)
    lstIndex = lst.index(res["eff"])
    refIndex = domObs[lstIndex]
    res["status"] = 0
    vec = np.zeros(n)
    vec[refIndex] = 1
    res["zOpt"] = vec
    res["iOpt"] = refIndex
    return res


##############
# Simulation #
##############

# Create a uniformly distributed random number between 0 and 1
def randomUniform():
    return np.random.rand()


# Create a random number between 0 and 1 with preference for numbers closer
# to zero. This type of sampling corresponds with picking floats from the
# list of all possible (computer) floats. This method is called "logarithmic"
# in the Maple procedure Generate(float).
def randomLog():
    return np.exp(np.random.uniform(-12, 0))


# Generate a random portfolio consisting of n assets. The two methods
# described above (randomUniform and randomLog) can be used for creating
# these random portfolios.
#
# Note that only the function name is passed, i.e., without round brackets.
def generatePortfolio(n, method):
    res = np.array([method() for i in range(n)])
    tot = np.sum(res)
    res = res / tot
    return res


# Create a list of nNonZeros distinct numbers between 0 and n-1. This list
# can be used as index list for making a selection of portfolios.
def createNonZeroList(n, nNonZeros):
    lst = [i for i in range(n)]
    shuffle(lst)
    return lst[:nNonZeros]


# Generate a random portfolio consisting of n assets. Only those assets
# with index present in nonZeroList are used in the portfolio composition.
# All other assets obtain a portfolio value of 0. The two selection methods
# are exactly those mentioned in the procedure generatePortolio.
def generatePortfolioWithZeros(n, nonZeroList, method):
    nNonZero = len(nonZeroList)
    portfolio = generatePortfolio(nNonZero, method)
    portfolioBis = np.zeros(n)
    for i in range(nNonZero):
        portfolioBis[nonZeroList[i]] = portfolio[i]
    return portfolioBis
