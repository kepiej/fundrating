import numpy as np
from scipy.optimize import linprog

#####################
# Manipulating data #
#####################

def rescaleMatrix(dataMatrix):
    avg = np.mean(dataMatrix, axis=0)
    return dataMatrix/avg

def extractXY(dataMatrix, listInOut):
    n = len(dataMatrix)
    m = len(dataMatrix[0])
    return dataMatrix[np.ix_([i for i in range(n)],listInOut)]

def extractDMU(dataMatrix, listDMU):
    n = len(dataMatrix)
    m = len(dataMatrix[0])
    res = dataMatrix[np.ix_(listDMU, [i for i in range(m)])]
    if len(listDMU) == 1:
        return res[0]
    else:
        return res

def createIndexList(header,idList):
    n = len(header)
    res = []
    for i in range(n):
        if header[i] in idList:
            res.append(i)
    return res

def extractTimeList(headerTime,headerDMU,idDMU):
    lst = createIndexList(headerDMU, [idDMU])
    res = []
    for i in lst:
        res.append(headerTime[i])
    res.sort()
    return res

def createUniqueIdList(header):
    res = list(set(header))
    res.sort()
    return res

def getDMUindex(headerTime,headerDMU,idTime,idDMU):
    n = len(headerTime)
    res = []
    for i in range(n):
        if (headerTime[i] == idTime) and (headerDMU[i] == idDMU):
            res.append(i)
    return res

def evalList(lst):
    res = []
    for el in lst:
        res.append(eval(el))
    return res



# Assume gInput <= 0
def dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput):
    n = len(x)
    p = len(x[0])
    q = len(y[0])
    rangeInput = [i for i in range(p)]
    rangeOutput = [i for i in range(q)]
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
    bnds = [(0, 1) for i in rangeDMU]
    bnds.append((-100, 1000))
    res = linprog(c, A_ub=aIneq, b_ub=bIneq, A_eq=aEq, b_eq=bEq,
                  bounds=bnds, method="interior-point")
    res.x = res.x[:n]
    res.fun = -res.fun
    resDict = {}
    resDict["eff"] = res.fun
    resDict["status"] = res.status
    resDict["zOpt"] = res.x
    resDict["iOpt"] = None
    return resDict

##################################
# Directional distance functions #
##################################

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

#################################
# Input and output efficiencies #
#################################

def inputEffDEAVRS(x,y,inputDMU,outputDMU):
    q = len(y[0])
    gInput = -np.array(inputDMU)
    gOutput = np.zeros(q)
    res = dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 - res["eff"]
    return res

def outputEffDEAVRS(x,y,inputDMU,outputDMU):
    p = len(x[0])
    gInput = np.zeros(p)
    gOutput = np.array(outputDMU)
    res = dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 + res["eff"]
    return res

def inputEffDEAVRS_sr(x,y,inputDMU,outputDMU,listFixedVars,listVariableVars):
    p = len(x[0])
    q = len(y[0])
    gInput = -np.array(inputDMU)
    for i in listFixedVars:
        gInput[i] = 0
    gOutput = np.zeros(q)
    res = dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 - res["eff"]
    return res

def inputEffFDHVRS_sr(x,y,inputDMU,outputDMU,listFixedVars,listVariableVars):
    p = len(x[0])
    q = len(y[0])
    gInput = -np.array(inputDMU)
    for i in listFixedVars:
        gInput[i] = 0
    gOutput = np.zeros(q)
    res = dirDistFDHVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 - res["eff"]
    return res

def inputEffFDHVRS(x,y,inputDMU,outputDMU):
    q = len(y[0])
    gInput = -np.array(inputDMU)
    gOutput = np.zeros(q)
    res = dirDistFDHVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 - res["eff"]
    return res

def outputEffFDHVRS(x,y,inputDMU,outputDMU):
    p = len(x[0])
    gInput = np.zeros(p)
    gOutput = np.array(outputDMU)
    res = dirDistFDHVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    res["eff"] = 1 + res["eff"]
    return res

