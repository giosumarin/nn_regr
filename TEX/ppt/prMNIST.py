import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=["ALL" ,"10%","15%","20%","25%","30%","35%","40%","45%","50%","55%","60%","65%","70%","75%","80%","85%","90%","95%"]
y=[98.34,98.34,98.34,98.34,98.35,98.36,98.35,98.36,98.36,98.36,98.32,98.30,98.33,98.34,98.34,98.38,98.31,98.27,98.28]
ax.set_ylabel("Accuratezza", color='red')
ax.set_xlabel("Pruning %")
ax.plot(x,y, '.-',label='first plot', color='red')
plt.grid()
#plt.ylim((98.2,98.4))
#plt.show()

ax2 = ax.twinx()

x=["ALL" ,"10%","15%","20%","25%","30%","35%","40%","45%","50%","55%","60%","65%","70%","75%","80%","85%","90%","95%"]
y=[None,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
ax2.plot(x,y, '.-',label='first plot',color='blue')
ax2.set_ylabel(r"$r_1$", color='blue')
#ax2.grid()
#plt.ylim((98.2,98.4))
plt.show()