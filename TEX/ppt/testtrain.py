import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=range(1,24)
y=[95.0967, 96.6683, 97.64, 98.3, 98.765, 98.9917, 99.2417, 99.3717, 99.5017, 99.62, 99.6783, 99.7433, 99.8017, 99.835, 99.8667, 99.895, 99.9233, 99.945, 99.9617, 99.975, 99.9783, 99.9867, 99.9933]
ax.set_ylabel("Accuratezza", color='red')
ax.set_xlabel("Epoca")
p1=ax.plot(x,y, '.-',label='first plot', color='red')
plt.grid()
#plt.xticks(rotation=70)
#plt.ylim((95,1300))
#plt.show()
y=[94.93, 96.38, 96.95, 97.34, 97.73, 97.77, 97.82, 97.81, 97.88, 97.97, 97.98, 98.07, 98.1, 98.06, 98.15, 98.24, 98.25, 98.25, 98.3, 98.32, 98.34, 98.34, 98.3]
ax.axhline(98.34, color='black', lw=1, alpha=0.5)
p2=ax.plot(x,y, '.-',label='first plot', color='orange')
#plt.title("Variazione dell’accuratezza della rete che ha appreso a classificare le cifre MNIST applicando la tecnica di weight sharing")
ax.legend((p1[0], p2[0]), ("Train","Test"))#, loc='upper center')
plt.show()
