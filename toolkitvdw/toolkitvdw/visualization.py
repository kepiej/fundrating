import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.colors import to_rgb
from scipy.spatial import ConvexHull

def init2d():
    fig = plt.figure()
    ax = fig.gca()
    return ax

def plot(ax, f, nGrid, xRange, **kwargs):
    xCo = np.linspace(xRange[0], xRange[1], nGrid)
    yCo = np.array([f(x) for x in xCo])
    ax.plot(xCo, yCo, **kwargs)

def point(ax, co, **kwargs):
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("linestyle", "None")
    ax.plot([co[0]], [co[1]], **kwargs)

def pointplot(ax, xCo, yCo, **kwargs):
    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "o")
    ax.plot(xCo, yCo, **kwargs)

def line(ax, co1, co2, **kwargs):
    kwargs.setdefault("linestyle", "solid")
    kwargs.setdefault("marker", "None")
    kwargs.setdefault("color", "black")
    ax.plot([co1[0], co2[0]], [co1[1], co2[1]], **kwargs)

def textplot(ax, co, text, **kwargs):
    point(ax, co, marker="None")
    ax.text(co[0], co[1], text, **kwargs)

def implicitplot(ax, f, nGrid, xRange, yRange, **kwargs):
    xCo = np.linspace(xRange[0], xRange[1], nGrid)
    yCo = np.linspace(yRange[0], yRange[1], nGrid)
    x, y = np.meshgrid(xCo, yCo)
    z = f(x, y)
    newKwargs = {}
    keys = kwargs.keys()
    if "color" in keys:
        newKwargs["colors"] = [kwargs["color"]]
    if "linestyle" in keys:
        newKwargs["linestyles"] = [kwargs["linestyle"]]
    if "linewidth" in keys:
        newKwargs["linewidths"] = [kwargs["linewidth"]]
    cp = ax.contour(x, y, z, [0], **newKwargs)
    if "label" in keys:
        cp.collections[0].set_label(kwargs["label"])

# Currently, this function only works for the default direction g.
# To be adapted in the future to accommodate for other directions.
def plotFrontierC(ax, xCo, yCo, boundBottom = None,
                  boundRight = None, g=[-1, 1], **kwargs):
    xCo = np.array(xCo)
    yCo = np.array(yCo)
    pts = np.column_stack((xCo, yCo))
    hull = ConvexHull(pts)
    ptsHull = pts[hull.vertices, :]
    n = len(ptsHull)
    ind = np.lexsort((-ptsHull[:, 1], -ptsHull[:, 0]))
    iTop = ind[0]
    iBottom = ind[-1]
    if iTop < iBottom:
        indNew = [i for i in range(iTop, iBottom + 1)]
    else:
        indNew = [i for i in range(iTop, n)] + [i for i in range(0, iBottom + 1)]
    kwargs.setdefault("linestyle", "solid")
    kwargs.setdefault("marker", "None")
    ax.plot(ptsHull[indNew, 0], ptsHull[indNew, 1], **kwargs)
    kwargs["label"] = None
    if boundBottom != None:
        line(ax, [ptsHull[iBottom, 0], boundBottom],
             [ptsHull[iBottom, 0], ptsHull[iBottom, 1]], **kwargs)
    if boundRight != None:
        line(ax, [ptsHull[iTop, 0], ptsHull[iTop, 1]],
             [boundRight, ptsHull[iTop, 1]], **kwargs)

def plotFrontierNC(ax, xCo, yCo, boundBottom = None,
                   boundRight = None, g=[-1, 1], **kwargs):
    n = len(xCo)
    xCo = np.array(xCo)
    yCo = np.array(yCo)
    ind = np.lexsort((yCo, xCo))
    xCo = xCo[ind]
    yCo = yCo[ind]
    xCoExt = []
    yCoExt = []
    for i in range(n - 1):
        if g[0] < 0:
            xCoExt.extend([xCo[i], xCo[i + 1], xCo[i + 1]])
            yCoExt.extend([yCo[i], yCo[i], yCo[i + 1]])
        else:
            xCoExt.extend([xCo[i], xCo[i], xCo[i + 1]])
            yCoExt.extend([yCo[i], yCo[i + 1], yCo[i + 1]])
    kwargs.setdefault("linestyle", "solid")
    kwargs.setdefault("marker", "None")
    ax.plot(xCoExt, yCoExt, **kwargs)
    kwargs["label"] = None
    if boundBottom != None:
        if g[0] < 0:
            line(ax, [xCo[0], boundBottom], [xCo[0], yCo[0]], **kwargs)
        else:
            line(ax, [xCo[-1], boundBottom], [xCo[-1], yCo[-1]], **kwargs)
    if boundRight != None:
        if g[0] > 0:
            line(ax, [xCo[0], yCo[0]], [boundRight, yCo[0]], **kwargs)
        else:
            line(ax, [xCo[-1], yCo[-1]], [boundRight, yCo[-1]], **kwargs)

def arrow(ax, coFrom, coTo, **kwargs):
    kwargs.setdefault("arrowstyle", "->")
    kwargs.setdefault("shrinkA", 0)
    kwargs.setdefault("shrinkB", 0)
    ax.annotate("", xy=coTo, xytext=coFrom,
                arrowprops=kwargs)

def plotCashflows(ax, tList, cList, tPos=True, cLength=False,
                  tLabels=[], cLabels=[], **kwargs):
    n = len(tList)
    iPos = 1
    xCo = []
    for i in range(n):
        if tPos:
            xCo.append(tList[i])
        else:
            xCo.append(iPos)
            iPos += 1
    yCo = []
    yAbs = np.abs(cList)
    yThreshold = np.max(yAbs)/2
    for i in range(n):
        if cLength:
            yCo.append(cList[i])
        else:
            if yAbs[i] < yThreshold:
                cVal = np.sign(cList[i])
            else:
                cVal = np.sign(cList[i]) * 2
            yCo.append(cVal)
    n = len(xCo)
    for i in range(n):
        coFrom = [xCo[i], 0]
        coTo = [xCo[i], yCo[i]]
        arrow(ax, coFrom, coTo, **kwargs)
    xLen = max(xCo) - min(xCo)
    xExtra = xLen / 10
    yLen = max(yCo) - min(yCo)
    yExtra = yLen / 100
    line(ax, [min(xCo) - xExtra, 0], [max(xCo) + xExtra, 0])
    pointplot(ax, xCo, [0 for i in range(n)], marker="None")
    pointplot(ax, xCo, yCo, marker="None")
    if len(tLabels) == 0:
        tLabels = [tList[i] for i in range(n)]
    for i in range(n):
        if cList[i] <= 0:
            va = "bottom"
            yCorrected = yExtra
        else:
            va = "top"
            yCorrected = -yExtra
        textplot(ax, [xCo[i], yCorrected], tLabels[i],
                 horizontalalignment="center",
                 verticalalignment=va)
    if len(cLabels) == 0:
        cLabels = [cList[i] for i in range(n)]
    for i in range(n):
        if cList[i] <= 0:
            va = "top"
            yCorrected = -yExtra
        else:
            va = "bottom"
            yCorrected = yExtra
        textplot(ax, [xCo[i], yCo[i] + yCorrected], cLabels[i],
                 horizontalalignment="center",
                 verticalalignment=va)

def init3d():
    fig = plt.figure()
    #ax = fig.gca(aspect="equal", projection="3d") # Old line to be replaced when using new version of Matplotlib
    ax = Axes3D(fig)
    return ax

# Currently not used
def box3d(ax, xRange, yRange, zRange, **kwargs):
    for i in range(2):
        ax.plot([xRange[0], xRange[1]],
                [yRange[0], yRange[0]],
                [zRange[i], zRange[i]], **kwargs)
        ax.plot([xRange[0], xRange[0]],
                [yRange[0], yRange[1]],
                [zRange[i], zRange[i]], **kwargs)
        ax.plot([xRange[1], xRange[1]],
                [yRange[1], yRange[0]],
                [zRange[i], zRange[i]], **kwargs)
        ax.plot([xRange[1], xRange[0]],
                [yRange[1], yRange[1]],
                [zRange[i], zRange[i]], **kwargs)
    ax.plot([xRange[0], xRange[0]],
            [yRange[0], yRange[0]],
            [zRange[0], zRange[1]], **kwargs)
    ax.plot([xRange[0], xRange[0]],
            [yRange[1], yRange[1]],
            [zRange[0], zRange[1]], **kwargs)
    ax.plot([xRange[1], xRange[1]],
            [yRange[0], yRange[0]],
            [zRange[0], zRange[1]], **kwargs)
    ax.plot([xRange[1], xRange[1]],
            [yRange[1], yRange[1]],
            [zRange[0], zRange[1]], **kwargs)

def plot3d(ax, f, nxGrid, nyGrid, xRange, yRange, **kwargs):
    kwargs.setdefault("color", "blue")
    xCo = np.linspace(xRange[0], xRange[1], nxGrid)
    yCo = np.linspace(yRange[0], yRange[1], nyGrid)
    x, y = np.meshgrid(xCo, yCo)
    z = f(x, y)
    if "zclip" in kwargs:
        cl = kwargs.pop("zclip")
        z = np.clip(z, cl[0], cl[1])
    light = LightSource(70, 135)
    white = np.ones((z.shape[0], z.shape[1], 3))
    illum_surf = light.shade_rgb(white * to_rgb(kwargs["color"]), z)
    lab = ""
    newKwargs = {}
    for el in kwargs:
        if el == "label":
            lab = kwargs[el]
        else:
            newKwargs[el] = kwargs[el]
    newKwargs.setdefault("rstride", 1)
    newKwargs.setdefault("cstride", 1)
    newKwargs.setdefault("facecolors", illum_surf)
    ax.plot([xCo[0]], [yCo[0]], linestyle="solid",
            color=kwargs["color"], label=lab)
    ax.plot_surface(x, y, z, **newKwargs)

def point3d(ax, co, **kwargs):
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("linestyle", "None")
    ax.plot([co[0]], [co[1]], [co[2]], **kwargs)

def pointplot3d(ax, xCo, yCo, zCo, **kwargs):
    newKwargs = {}
    for el in kwargs:
        if el == "markersize":
            newKwargs["s"] = kwargs[el]
        else:
            newKwargs[el] = kwargs[el]
    ax.scatter(xCo, yCo, zCo, **newKwargs)

def textplot3d(ax, co, text, **kwargs):
    point3d(ax, co, marker="None")
    ax.text(co[0], co[1], co[2], text, **kwargs)

def line3d(ax, co1, co2, **kwargs):
    ax.plot([co1[0], co2[0]], [co1[1], co2[1]], [co1[2], co2[2]],
            **kwargs)

def curve3d(ax, xCo, yCo, zCo, **kwargs):
    n = len(xCo)
    for i in range(1, n):
        co1 = [xCo[i-1], yCo[i-1], zCo[i-1]]
        co2 = [xCo[i], yCo[i], zCo[i]]
        line3d(ax, co1, co2, **kwargs)
        kwargs.pop("label", None)