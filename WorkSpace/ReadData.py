import numpy as np
import matplotlib.pyplot as plt
import xlrd

data=xlrd.open_workbook('Load1997.xls')

sheet1=data.sheet_by_index(0)
sheet2=data.sheet_by_index(1)
loaddata=[]
otherdata=[]
maindata=[]
maindata1=[]

for i in range(1,366):
    rowdata1=sheet1.row_values(i)
    rowdata2=sheet2.row_values(i)
    loaddata.append(rowdata1[3:])
    otherdata.append(rowdata2[4:])


for day in range(2,365):
    for t in range(0,48):
        maindata.append(otherdata[day-2]+[loaddata[day-2][t]]+
                        otherdata[day-1]+[loaddata[day-1][t]]+
                        otherdata[day]+[loaddata[day][t]])

np.save('maindata.npy',maindata)

for day in range(3,365):
    for t in range(0,48):
        if t==0:
            maindata1.append(otherdata[day-2]+[loaddata[day-3][47]]+[loaddata[day-2][t]]+
                            otherdata[day-1]+[loaddata[day-2][47]]+[loaddata[day-1][t]]+
                            otherdata[day]+[loaddata[day][t]])
        else:
            maindata1.append(otherdata[day-2]+[loaddata[day-2][t-1]]+[loaddata[day-2][t]]+
                            otherdata[day-1]+[loaddata[day-1][t-1]]+[loaddata[day-1][t]]+
                            otherdata[day]+[loaddata[day][t]])

np.save('maindata1.npy',maindata1)

'''
X_plot=np.linspace(0.5,24,48)

plt.figure()
lw=2
for i in range(0,7):
    plt.plot(X_plot,loaddata[i],lw=lw,
    label='Day %s' % i)
    
plt.xlabel('Time')
plt.ylabel('Load')
plt.xlim(0,24)
plt.title('One Week Load')
plt.legend()
plt.show()
'''


