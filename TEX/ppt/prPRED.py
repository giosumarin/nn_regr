import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=["ALL","10%","20%","30%","40%","50%","60%","70%","80%","90%"]
y=[1270,708,708,708,707,707,707,706,700,660]
ax.set_ylabel(r'$\epsilon$', color='red')
ax.plot(x,y, '.-',label='first plot', color='red')
ax.set_xlabel("Pruning %")
plt.grid()
plt.ylim((650,1300))
#plt.show()
#plt.title("Variazione dell’accuratezza della rete che ha appreso a classificare le cifre MNIST applicando la tecnica di weight sharing")
ax2 = ax.twinx()

x=["ALL","10%","20%","30%","40%","50%","60%","70%","80%","90%"]
#plt.ylim((.48,.535))
y=[None,1.80,1.60,1.40,1.20,1.00,0.81,0.61,0.41,0.21]
ax2.plot(x,y, '.-',label='first plot',color='blue')
#ax2.bar(x,y,label='first plot',color='blue', width=0.4,alpha=0.5)
ax2.set_ylabel(r'$r_1$', color='blue')
#ax2.grid()
#plt.ylim((.48,.535))
#plt.grid()

plt.show()
#plt.savefig('png/pr_pred.png')