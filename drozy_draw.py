import matplotlib.pyplot as plt
import re
import math
import numpy as np

dataY = []

flieTxt = open('./test/drozy_1-1.txt', 'r')
lineOfAll = flieTxt.readlines()
flieTxt.close()

for line in lineOfAll:
    pattern = r'eye_lip_distance:(\d+.\d+)'
    matches = re.findall(pattern, line.strip())
    dataY.extend(matches)
    dataY = list(map(float, dataY))

lenX = len(dataY)
dataX = []

for i in range(lenX):
    dataX.append(i)

#print(f'data:{dataY}')

def baselineBn(dataY):
    dataY_b = []
    dataY_b.append(dataY[0])
    Alpha = [0] * lenX
    for i in range(1, lenX):
        Alpha[i] = smoothingFactor(dataY_b, i)
        bI = (1 - Alpha[i]) * dataY_b[i - 1] + Alpha[i] * dataY[i]
        #print(f'bI{bI}')
        dataY_b.append(round(bI,4))
    #print(f'bn:{dataY_b}')
    return dataY_b


def smoothingFactor(dataY_b, n):
    a0 = 0.4
    ad = 15
    aa = 0.5
    ab = 2
    am = 0.7
    exp1 = (-1) * ad * ((dataY[n] - dataY[n - 1]) ** 2)
    exp_1 = math.exp(exp1)
    exp2 = (-1) * aa * (dataY[n] - dataY_b[n - 1]) if dataY[n] - dataY_b[n - 1] > 0 else 0
    exp_2 = math.exp(exp2)
    exp3 = (-1) * ab * (dataY_b[n - 1] - dataY[n]) if dataY_b[n - 1] - dataY[n] > 0 else 0
    exp_3 = math.exp(exp3)
    exp4 = dataY[n] - am * getMedian(dataY, n)
    exp_4 = 1 if exp4 >= 0 else 0
    return a0 * exp_1 * exp_2 * exp_3 * exp_4


def getMedian(dataY, n):
    d = []
    d.extend(dataY[1:n+1])
    d.sort()
    if len(d) % 2 == 0:
        return (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
    else:
        return d[len(d) // 2]


dataY_b = baselineBn(dataY)
plt.figure(figsize=(15, 8))
plt.plot(dataX, dataY, color='black', marker= '',linestyle='-', alpha=0.5, linewidth=1, label='eyeLipDistance')
plt.plot(dataX, dataY_b, color='red', linestyle='-', alpha=0.8, linewidth=1.5, label='baseline')
plt.legend()
plt.xlabel('frame')
plt.ylabel('millimeter')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show()

data_normalized=[]
for i in range(lenX):
    opening = dataY[i] / dataY_b[i]
    if opening<=0.6:
        opening=0
    elif opening<0.8:
        opening*=opening
    elif opening>1.05:
        opening=opening**0.5
    data_normalized.append(round(opening,4))
plt.figure(figsize=(15,4))
plt.plot(dataX,data_normalized,color='green',linestyle='-', alpha=1, linewidth=2)
plt.xlabel('Time[frame]')
plt.ylabel('Normalized Eye Opening')
plt.ylim(ymin=0)
plt.show()

