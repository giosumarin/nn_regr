import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=["ALL","1638\n6553\n89", "3276\n13107\n115", "4915\n19660\n140", "6553\n26214\n166"]
y=[1270,771,678,709,675]
ax.set_ylabel("Accuratezza", color='red')
ax.set_xlabel("Clusters")
ax.plot(x,y, '.-',label='first plot', color='red')
plt.grid()
#plt.ylim((93.5,98.5))
#plt.show()
#plt.title("Variazione dell’accuratezza della rete che ha appreso a classificare le cifre MNIST applicando la tecnica di weight sharing")
ax2 = ax.twinx()

x=["ALL","1638\n6553\n89", "3276\n13107\n115", "4915\n19660\n140", "6553\n26214\n166"]
y=[None,0.6,0.7,0.8,0.9]
ax2.plot(x,y, '.-',label='first plot',color='blue')
#ax2.bar(x,y,label='first plot',color='blue', widt"Accuracy"h=0.4,alpha=0.5)
ax2.set_ylabel(r'$r_2$', color='blue')
#ax2.grid()
#plt.ylim((.48,.535))
#plt.grid()
plt.show()
#plt.savefig('png/ws_pred.png')