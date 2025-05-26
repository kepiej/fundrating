import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.colors as mcolors


###########################
# Miscellaneous functions #
###########################

class StructureCSV:
    def __init__(self):
        self.left = None
        self.top = None
        self.body = None


def secondsToMinSec(seconds):
    m, s = divmod(int(seconds), 60)
    return "{0:d} minutes, {1:d} seconds".format(m, s)


def secondsToHourMinSec(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return "{0:d} hours, {1:d} minutes, {2:d} seconds".format(h, m, s)


def checkNumeric(a, type, b):
    epsilon = 1E-06
    if type == "==":
        return abs(a - b) <= epsilon
    elif type == "<=":
        return a <= b + epsilon
    elif type == "<":
        return a < b + epsilon
    elif type == ">=":
        return a >= b - epsilon
    elif type == ">":
        return a > b - epsilon
    else:
        return False


def readCSVfile(filename, separationChar=",", quoteChar='"', nSkipRows=0, nSkipCols=0,
                nHeadersTop=1, nHeadersLeft=1):

    def splitWithQuoteChar(input):
        lst = []
        wrd = ""
        flagQuote = False
        for letter in input:
            if letter == quoteChar:
                flagQuote = not flagQuote
            elif flagQuote:
                wrd += letter
            else:
                if letter == separationChar:
                    lst.append(wrd.strip())
                    wrd = ""
                else:
                    wrd += letter
        lst.append(wrd.strip())
        return lst

    f = open(filename, "r")
    listIn = list(f)
    f.close()
    for i in range(nSkipRows):
        listIn.pop(0)
    listHeaderTop = []
    listHeaderLeft = [[] for i in range(nHeadersLeft)]
    for i in range(nHeadersTop):
        header = splitWithQuoteChar(listIn.pop(0))
        header = header[nSkipCols + nHeadersLeft:]
        listHeaderTop.append(header)
    nRows = len(listIn)
    body = []
    for i in range(nRows):
        line = splitWithQuoteChar(listIn[i])
        line = line[nSkipCols:]
        for j in range(nHeadersLeft):
            listHeaderLeft[j].append(line.pop(0))
        body.append(line)
    res = StructureCSV()
    res.left = listHeaderLeft
    res.top = listHeaderTop
    res.body = body
    return res


def writeCSVfile(matrix, filename, separationChar=","):
    f = open(filename, "w")
    n = len(matrix)
    m = len(matrix[0])
    for i in range(n):
        for j in range(m - 1):
            print(matrix[i][j], end=separationChar, file=f)
        print(matrix[i][m - 1], file=f)
    f.close()


def writeCofile(xCo, yCo, zCo, filename, color="red"):
    f = open(filename, "w")
    n = len(xCo)
    col = "0x" + mcolors.to_hex(color).lstrip("#")
    col = eval(col)
    print("compact=0", file=f)
    print("dem=False", file=f)
    print("data=", file=f)
    for i in range(n):
        print(xCo[i], end="\t", file=f)
        print(yCo[i], end="\t", file=f)
        print(zCo[i], end="\t", file=f)
        print(-1, end="\t", file=f)
        print(col, file=f)
    f.close()


def evalToMatrix(listOflist):
    n = len(listOflist)
    m = len(listOflist[0])
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            try:
                res[i, j] = eval(listOflist[i][j])
            except:
                res[i, j] = np.nan
    return res
