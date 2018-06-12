import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd

def plot7day(start, loaddata):
    X_plot = np.linspace(0.5, 24, 48)
    daylist=['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
    plt.figure()
    lw = 2
    for i in range(0, 7):
        plt.plot(X_plot, loaddata[start + i], lw=lw, label=daylist[i])

    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.xlim(0, 24)
    plt.title('One Week Load')
    plt.legend()
    plt.show()

data1 = xlrd.open_workbook('Load1997.xls')
data2 = xlrd.open_workbook('Load1998.xls')
data3 = xlrd.open_workbook('Load1999.xls')

sheet1_1 = data1.sheet_by_index(0)
sheet1_2 = data1.sheet_by_index(1)

sheet2_1 = data2.sheet_by_index(0)
sheet2_2 = data2.sheet_by_index(1)

sheet3_1 = data3.sheet_by_index(0)
sheet3_2 = data3.sheet_by_index(1)

loaddata = []
otherdata = []
maindata = []

for i in range(1, sheet1_1.nrows):
    rowdata1 = sheet1_1.row_values(i)
    rowdata2 = sheet1_2.row_values(i)
    loaddata.append(rowdata1[3:])
    otherdata.append(rowdata2[4:])

for i in range(1, sheet2_1.nrows):
    rowdata1 = sheet2_1.row_values(i)
    rowdata2 = sheet2_2.row_values(i)
    loaddata.append(rowdata1[3:])
    otherdata.append(rowdata2[4:])

for i in range(1, sheet3_1.nrows):
    rowdata1 = sheet3_1.row_values(i)
    rowdata2 = sheet3_2.row_values(i)
    loaddata.append(rowdata1[3:])
    otherdata.append(rowdata2[4:])

for day in range(3, len(loaddata)):
    for t in range(0, 48):
        if t == 0:
            maindata.append(
                otherdata[day - 2] + [loaddata[day - 3][47]] + [loaddata[day - 2][t]] +
                otherdata[day - 1] + [loaddata[day - 2][47]] + [loaddata[day - 1][t]] +
                otherdata[day] + [loaddata[day][t]])
        else:
            maindata.append(
                otherdata[day - 2] + [loaddata[day - 2][t - 1]] + [loaddata[day - 2][t]] +
                otherdata[day - 1] + [loaddata[day - 1][t - 1]] + [loaddata[day - 1][t]] +
                otherdata[day] + [loaddata[day][t]])

data_train = maindata[:-31 * 48]
data_test = maindata[-31 * 48:]

# X_plot = np.linspace(0.5, 24, 48)
# daylist=['1997-1-28 -10°C','1997-3-10 7.6°C','1997-5-26 10.3°C','1997-6-9 20.5°C','1997-6-30 25.3°C','1997-11-3 3.6°C']
# plt.figure()
# lw = 2

# plt.plot(X_plot, loaddata[27], lw=lw, label=daylist[0])
# plt.plot(X_plot, loaddata[68], lw=lw, label=daylist[1])
# plt.plot(X_plot, loaddata[145], lw=lw, label=daylist[2])
# plt.plot(X_plot, loaddata[159], lw=lw, label=daylist[3])
# plt.plot(X_plot, loaddata[180], lw=lw, label=daylist[4])
# plt.plot(X_plot, loaddata[306], lw=lw, label=daylist[5])

# plt.xlabel('Time')
# plt.ylabel('Load')
# plt.xlim(0, 24)
# plt.legend()
# plt.show()

np.save('AllData.npy', maindata)
np.save('Data-Train.npy', data_train)
np.save('Data-Test.npy', data_test)

def outtxt(result):
    result = open('result.txt', 'w')

    df = pd.DataFrame(maindata)

    print("前五个数据")
    print(df.head(), file=result)

    print("简单统计描述")
    print(df.describe(), file=result)

    print("协方差矩阵")
    print(df.cov(), file=result)

    print("相关系数矩阵")
    print(df.corr(), file=result)

    result.close()



