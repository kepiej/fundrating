from toolkitvdw.visualization import *
from toolkitvdw.finance import *

# Test DEAVRS

np.set_printoptions(precision=4, suppress=True)
x = np.array([[161.17, 433.2, 535.8, 9744, 6.9, 0.63, 19.65],
              [172.96, 586.8, 569.4, 7222, 18, 0.33, 14.58],
              [2613.43, 5808.7, 8857, 51828, 172.6, 7.29, 262.44],
              [964.04, 1423.2, 3824.9, 26075, 21.4, 1.45, 82.58],
              [605.42, 1042.5, 5838.2, 28054, 23, 0.77, 72.41],
              [921.45, 1090.5, 3627, 79739, 48.2, 3.16, 113.04],
              [616.17, 773.2, 4067.4, 32476, 22.4, 1.84, 107.61],
              [959.7, 1285.4, 9030.9, 69907, 73.3, 2.69, 121.8],
              [242.6, 158.7, 552.2, 7741, 5.3, 0.83, 13.08],
              [2750.4, 2499.7, 7966.8, 56647, 72.9, 9.66, 322.68]])

x = x[:, [0, 6]]
print(x)

y = np.array([[111.5150071],
              [147.8320363],
              [1324.219725],
              [444.6712165],
              [571.8401438],
              [674.6517768],
              [443.8015634],
              [785.6400245],
              [146.9282382],
              [1448.45125]])

n = len(x)
resList = []
for i in range(n):
    inputDMU = x[i]
    outputDMU = y[i]
    gInput = x[i]
    gOutput = y[i]
    res = dirDistDEAVRS(x, y, inputDMU, outputDMU, gInput, gOutput)
    resList.append(res["eff"])
print(np.array(resList))

# Test visualization

n = 10
xRange = [-2, 2]
yRange = [-2, 2]
xCo = []
yCo = []
zCo = []
for i in range(n):
    for j in range(n):
        x = xRange[0] + (xRange[1] - xRange[0]) / (n - 1) * i
        y = yRange[0] + (yRange[1] - yRange[0]) / (n - 1) * j
        z = 0
        xCo.append(x)
        yCo.append(y)
        zCo.append(z)
fig3d = init3d()
pointplot3d(fig3d, xCo, yCo, zCo, color="orange", marker="o",
            markersize=15, label="Orange grid")


def f(x, y):
    return -1 / 4 * (x ** 2 + y ** 2) + 3


xCof = []
yCof = []
zCof = []
for i in range(n):
    for j in range(n):
        x = xRange[0] + (xRange[1] - xRange[0]) / (n - 1) * i
        y = yRange[0] + (yRange[1] - yRange[0]) / (n - 1) * j
        z = f(x, y)
        xCof.append(x)
        yCof.append(y)
        zCof.append(z)

pointplot3d(fig3d, xCof, yCof, zCof, color="blue", marker="o",
            markersize=20, label="Blue dots")
nList = len(xCo)
for i in range(nList):
    line3d(fig3d, [xCo[i], yCo[i], zCo[i]],
           [xCof[i], yCof[i], zCof[i]], linestyle="solid",
           color="gray", alpha=0.5)
fig3d.set_xlabel("x")
fig3d.set_ylabel("y")
fig3d.set_zlabel("z")
fig3d.view_init(azim=50, elev=13)
plt.legend(loc="upper right")
plt.show()
