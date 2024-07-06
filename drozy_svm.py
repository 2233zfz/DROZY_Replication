from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
import csv
from datetime import datetime,timedelta
import numpy as np
import random
import matplotlib.pyplot as plt

W = 0.5  # 惯性因子
c1 = 0.2  # 学习因子
c2 = 0.5  # 学习因子
n_iterations = 5  # 迭代次数
n_particles = 50  # 种群规模

def StringToSeconds(time_str):
    parts = time_str.split(":")
    hours, minutes, seconds = map(float, parts)
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def fitness_function(position):
    clf = svm.SVC(kernel='rbf', C=position[0], gamma=position[1])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    # print(f"tn:{tn} fn:{fn}")
    # print(f"fp:{fp} tp:{tp}")
    return tp+tn,fp+fn

def Pso(W,c1,c2,n_iterations,n_particles):
    particle_position_vector = np.array(
        [np.array([random.random() * 10, random.random() * 10]) for _ in range(n_particles)])
    pbest_position = particle_position_vector  # 个体极值等于最初位置
    pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])  # 个体极值的适应度值
    gbest_fitness_value = np.array([float('inf'), float('inf')])  # 全局极值的适应度值
    gbest_position = np.array([float('inf'), float('inf')])
    velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])  # 粒子速度
    # 迭代更新
    iteration = 0
    cadidateCurve=[]
    while iteration < n_iterations:
        print(f"iteration={iteration}")
        for i in range(n_particles):  # 对每个粒子进行循环
            fitness_cadidate = fitness_function(particle_position_vector[i])  # 每个粒子的适应度值=适应度函数（每个粒子的具体位置）
            cadidateCurve.append(fitness_cadidate[1])
            # print(f"0:{fitness_cadidate[0]} 1:{fitness_cadidate[1]}")
            if (pbest_fitness_value[i] > fitness_cadidate[
                1]):  # 每个粒子的适应度值与其个体极值的适应度值(pbest_fitness_value)作比较，如果更优的话，则更新个体极值，
                pbest_fitness_value[i] = fitness_cadidate[1]
                pbest_position[i] = particle_position_vector[i]

            if (gbest_fitness_value[1] > fitness_cadidate[
                1]):  # 更新后的每个粒子的个体极值与全局极值(gbest_fitness_value)比较，如果更优的话，则更新全局极值
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]

            elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]

        for i in range(n_particles):  # 更新速度和位置，更新新的粒子的具体位置
            new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position
        iteration = iteration + 1
    plt.figure(figsize=(10, 10))
    plt.plot([i for i in range(n_iterations*n_particles)], cadidateCurve, color='green', marker='', linestyle='-', alpha=0.5, linewidth=1, label='fitness_cadidate_curve')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('fitness')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.show()
    return gbest_position


datadir=['./svm_data/csv_data/csv_data_1-1.csv','./svm_data/csv_data/csv_data_1-2.csv','./svm_data/csv_data/csv_data_1-3.csv',
         './svm_data/csv_data/csv_data_2-1.csv','./svm_data/csv_data/csv_data_2-2.csv','./svm_data/csv_data/csv_data_2-3.csv',
         './svm_data/csv_data/csv_data_3-1.csv','./svm_data/csv_data/csv_data_3-2.csv','./svm_data/csv_data/csv_data_3-3.csv',
         './svm_data/csv_data/csv_data_4-1.csv','./svm_data/csv_data/csv_data_4-2.csv','./svm_data/csv_data/csv_data_4-3.csv',
         './svm_data/csv_data/csv_data_5-1.csv','./svm_data/csv_data/csv_data_5-2.csv','./svm_data/csv_data/csv_data_5-3.csv',
         './svm_data/csv_data/csv_data_6-1.csv','./svm_data/csv_data/csv_data_6-2.csv','./svm_data/csv_data/csv_data_6-3.csv',
        './svm_data/csv_data/csv_data_7-2.csv','./svm_data/csv_data/csv_data_7-3.csv','./svm_data/csv_data/csv_data_8-1.csv',
         './svm_data/csv_data/csv_data_8-2.csv','./svm_data/csv_data/csv_data_8-3.csv','./svm_data/csv_data/csv_data_9-2.csv',
'./svm_data/csv_data/csv_data_9-3.csv','./svm_data/csv_data/csv_data_10-1.csv','./svm_data/csv_data/csv_data_10-3.csv',
'./svm_data/csv_data/csv_data_11-1.csv','./svm_data/csv_data/csv_data_11-2.csv'
         ]

data=[]
labels=[]
flag1=0
flag0=0
for i in range(len(datadir)):
    with open(datadir[i]) as csvfile:
        reader = csv.reader(csvfile)
        counter=0
        for row in reader:
            if counter==0:
                counter += 1
                continue
            data.append([float(tk) for tk in row[:-1]])
            groundtruth=StringToSeconds(row[-1])
            if float(groundtruth)>=0.5:
                labels.append(1)
                flag1+=1
            else:
                labels.append(-1)
                flag0+=1

print(f"样本数量：{len(labels)}")
print(f"正例数量：{flag1}")
print(f"负例数量：{flag0}")
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=10)
# X_train=data
# X_test=data
# y_train=labels
# y_test=labels  ##所有数据都作为训练集和数据集
bestposition=Pso(W,c1,c2,n_iterations,n_particles)
print(f"C={bestposition[0]} gamma={bestposition[1]}")
clf = svm.SVC(kernel='rbf', C=bestposition[0], gamma=bestposition[1])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred,average='binary',zero_division=1))
print('Recall: ', recall_score(y_test, y_pred,average='binary',zero_division=1))
print(f"tn:{tn} fn:{fn}")
print(f"fp:{fp} tp:{tp}")





