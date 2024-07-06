import sys
sys.path.append("./FaceAlignment")
import csv
from datetime import datetime,timedelta


def getBlinkParameters(blinkClassifier,blinkCount,microsleepCount,W,i):
    j=W*30+i
    if j>len(blinkClassifier):
        j=len(blinkClassifier)
    blinkNumber = 0
    blinkFrameCount=0
    p1,p2,p3=0,0,0
    for k in range(i,j):
        if blinkCount[k]==1:
            blinkNumber+=1
        if blinkClassifier[k]==1:
            p1+=1
            blinkFrameCount+=1
        elif blinkClassifier[k]==2:
            p2+=1
            blinkFrameCount += 1
        elif blinkClassifier[k]==3:
            p3+=1
            blinkFrameCount += 1
        else:
            continue
    microsleep=0
    for k in range(j,i):
        if microsleepCount[k]>=10:
            microsleep+=1
    if blinkNumber==0:
        return 0,0,0,0,microsleep
    return (p1+p2+p3)/blinkNumber,(p1+p2)/blinkNumber,p2/blinkNumber,(p2+p3)/blinkNumber,microsleep

def getHistogramParameters(data_normalized,W,i):
    j=W*30+i
    if j>len(blinkClassifier):
        j=len(blinkClassifier)
    p1,p2,p3,p4,p5,p6,p7,p8,p9,p10=0,0,0,0,0,0,0,0,0,0
    for k in range(i,j):
        if 0<=float(data_normalized[k])<0.1:
            p1+=1
        elif 0.1<=float(data_normalized[k])<0.2:
            p2+=1
        elif 0.2<=float(data_normalized[k])<0.3:
            p3+=1
        elif 0.3<=float(data_normalized[k])<0.4:
            p4+=1
        elif 0.4<=float(data_normalized[k])<0.5:
            p5+=1
        elif 0.5<=float(data_normalized[k])<0.6:
            p6+=1
        elif 0.6<=float(data_normalized[k])<0.7:
            p7+=1
        elif 0.7<=float(data_normalized[k])<0.8:
            p8+=1
        elif 0.8<=float(data_normalized[k])<0.9:
            p9+=1
        elif 0.9<=float(data_normalized[k]):
            p10+=1
    base=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
    return p1/base,p2/base,p3/base,p4/base,p5/base,p6/base,p7/base,p8/base,p9/base,p10/base

def get15Parameters(blinkClassifier,blinkCount,microsleepCount,data_normalized,W,flag):
    data=[]
    b1,b2,b3,b4,b5=getBlinkParameters(blinkClassifier,blinkCount,microsleepCount,W,flag)
    h1,h2,h3,h4,h5,h6,h7,h8,h9,h10=getHistogramParameters(data_normalized,W,flag)
    data.extend([b1, b2, b3, b4, b5])
    data.extend([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10])
    return data

blinkClassifier = []
blinkCount = []
microsleepCount = []
data_normalized= []

with open('./svm_data/txt_data/blinkClassifier_11-2', 'r') as f:
    for line in f:
        data = line.strip().split()
        if len(data) == 4:
            blinkClassifier.append(int(data[0]))
            blinkCount.append(int(data[1]))
            microsleepCount.append(int(data[2]))
            data_normalized.append(data[3])


rtValid=[]
with open('D:/DROZY_and_NTHU/GazeNormalization-cpu_1/testpart/DROZY/pvt-rt/11-2.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')
    counterCsv = -1
    startTime=next(csv_reader)
    startTime = datetime.strptime(startTime[0], '%Y-%m-%d_%H.%M.%S.%f')
    for row in csv_reader:
        if counterCsv == -1:
            counterCsv += 1
            continue
        else:
            time1 = row[0]
            time2 = row[1]
            time1 = datetime.strptime(time1, '%Y-%m-%d_%H.%M.%S.%f')
            time2 = datetime.strptime(time2, '%Y-%m-%d_%H.%M.%S.%f')
            time_difference = time2 - time1
        if time1-startTime>timedelta(minutes=1):
            rtValid.append((time1,time_difference))
        counterCsv+=1
timeStamps=[]

with open(r'D:\DROZY_and_NTHU\GazeNormalization-cpu_1\testpart\DROZY\timestamps\11-2.txt',"r") as txt:
    for line in txt:
        parts = line.split()
        line = " ".join(parts[:-1])
        timeForFrame = datetime.strptime(line, '%Y %m %d %H %M %S %f')
        timeStamps.append(timeForFrame)
    txt.close()

header=[]
for i in range(1, 106):
    header.extend([f'p{i}'])
header.append('groundtruth')

with open("./svm_data/csv_data/csv_data_11-2.csv","w",newline="") as csv1:
    writer = csv.writer(csv1)
    writer.writerow(header)

    for rt in rtValid:
        flag30, flag35, flag40, flag45, flag50, flag55, flag60 = 0, 0, 0, 0, 0, 0, 0
        c1, c2, c3, c4, c5, c6, c7 = 0, 0, 0, 0, 0, 0, 0
        for ts in timeStamps:
            if rt[0] - ts < timedelta(seconds=60) and c1 == 0:
                flag60 = timeStamps.index(ts)
                c1 = 1
            if rt[0] - ts < timedelta(seconds=55) and c2 == 0:
                flag55 = timeStamps.index(ts)
                c2 = 1
            if rt[0] - ts < timedelta(seconds=50) and c3 == 0:
                flag50 = timeStamps.index(ts)
                c3 = 1
            if rt[0] - ts < timedelta(seconds=45) and c4 == 0:
                flag45 = timeStamps.index(ts)
                c4 = 1
            if rt[0] - ts < timedelta(seconds=40) and c5 == 0:
                flag40 = timeStamps.index(ts)
                c5 = 1
            if rt[0] - ts < timedelta(seconds=35) and c6 == 0:
                flag35 = timeStamps.index(ts)
                c6 = 1
            if rt[0] - ts < timedelta(seconds=30) and c7 == 0:
                flag30 = timeStamps.index(ts)
                c7 = 1
                break
        data30 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 30, flag30)
        data35 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 35, flag35)
        data40 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 40, flag40)
        data45 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 45, flag45)
        data50 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 50, flag50)
        data55 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 55, flag55)
        data60 = get15Parameters(blinkClassifier, blinkCount, microsleepCount, data_normalized, 60, flag60)
        data106 = data30 + data35 + data40 + data45 + data50 + data55 + data60
        data106.append(rt[1])
        writer.writerow(data106)
print('csv文件已保存')

