import numpy as np
from scipy.optimize import minimize, brentq
from toolkitvdw.miscellaneous import *


# Solve the equation f(x) == 0 for x between a and b
# Example:
# ry = 0.05
# def eq(rm):
#     return (1+ry)-(1+rm)**12
# print(fsolve(eq, 0, 1))
def fsolve(f, a, b):
    if checkNumeric(f(a), "==", 0):
        return a
    elif checkNumeric(f(b), "==", 0):
        return b
    elif f(a) * f(b) < 0:
        return brentq(f, a, b)
    else:
        n = 100
        for i in range(n):
            xi = a + i * (b - a) / n
            if checkNumeric(f(xi), "==", 0):
                return xi
            elif f(xi) * f(b) < 0:
                return brentq(f, xi, b)
    return np.nan


# Find the integer number between the integer numbers na and nb
# where the function value of f changes sign. It is assumed
# that f(na)*f(nb) < 0.
def intZero(f, na, nb):
    if na == nb:
        return na
    elif na + 1 == nb:
        return nb
    else:
        nmid = (na + nb) // 2
        if checkNumeric(f(nmid), "==", 0):
            return nmid
        elif f(nmid) * f(na) < 0:
            return intZero(f, na, nmid)
        else:
            return intZero(f, nmid, nb)


def solveNLP(obj, x0, minimization=True, output=False, nIter=100,
             **kwargs):
    def f(x):
        if minimization == False:
            return -obj(x)
        else:
            return obj(x)

    constr = []
    listEq = []
    listGt = []
    if "eqcon" in kwargs:
        listEq = kwargs["eqcon"]
        nEq = len(listEq)
        for i in range(nEq):
            constr.append({'type': 'eq', 'fun': listEq[i]})
    if "gtcon" in kwargs:
        listGt = kwargs["gtcon"]
        nGt = len(listGt)
        for i in range(nGt):
            constr.append({'type': 'ineq', 'fun': listGt[i]})
    bnds = []
    if "bounds" in kwargs:
        bnds = kwargs["bounds"]
    if (constr == []) and (bnds == []):
        res = minimize(f, x0, options={'disp': output})
    elif constr == []:
        res = minimize(f, x0, bounds=bnds, options={'disp': output,
                                                    'maxiter': nIter})
    elif bnds == []:
        res = minimize(f, x0, constraints=constr,
                       options={'disp': output, 'maxiter': nIter})
    else:
        res = minimize(f, x0, constraints=constr, bounds=bnds,
                       options={'disp': output, 'maxiter': nIter})
    if minimization == False:
        res.fun = -res.fun
    # Check bounds and constraints:
    xOpt = res["x"]
    n = len(xOpt)
    # Check bounds
    problem = False
    if bnds != []:
        for i in range(n):
            if (bnds[i][0] == None) and (bnds[i][1] != None):
                if xOpt[i] > bnds[i][1]:
                    problem = True
            if (bnds[i][0] != None) and (bnds[i][1] == None):
                if xOpt[i] < bnds[i][0]:
                    problem = True
            if (bnds[i][0] != None) and (bnds[i][1] != None):
                if (xOpt[i] > bnds[i][1]) or (xOpt[i] < bnds[i][0]):
                    problem = True
            if problem:
                res["status"] = -1
                res["success"] = False
                res["message"] = "Optimal solution violates bounds"
                return res
    # Check equality constraints:
    for f in listEq:
        problem = not(checkNumeric(f(xOpt), "==", 0))
        if problem:
            res["status"] = -1
            res["success"] = False
            res["message"] = "Optimal solution violates equality constraints"
            return res
    # Check inequality constraints:
    for f in listGt:
        problem = not (checkNumeric(f(xOpt), ">=", 0))
        if problem:
            res["status"] = -1
            res["success"] = False
            res["message"] = "Optimal solution violates inequality constraints"
            return res
    return res

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


def solveNLPglobal(obj, x0, minimization=True, output=False, nIter=100, 
                   nRestart=10, **kwargs):
    n = len(x0)
    sList = []
    bestOpt = None
    wrongOpt = None
    # for i in range(n):
    #     s = [0 for i in range(n)]
    #     s[i] = 1
    #     sList.append(s)
    for i in range(nRestart):
        s = generatePortfolio(n, randomUniform)
        sList.append(s)
    for s in sList:
        res = solveNLP(obj, s, minimization=minimization, output=output,
                       nIter=nIter, **kwargs)
        if res["success"] == True:
            if bestOpt == None:
                bestOpt = res
            else:
                if minimization:
                    if res.fun < bestOpt.fun:
                        bestOpt = res
                else:
                    if res.fun > bestOpt.fun:
                        bestOpt = res
        else:
            wrongOpt = res
    if bestOpt != None:
        return bestOpt
    else:
        return wrongOpt

