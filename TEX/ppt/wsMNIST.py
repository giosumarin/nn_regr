import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=["ALL"    ,"16\n8"   ,"32\n8"   ,"32\n16"  ,"64\n16"  ,"64\n32"  ,"128\n32","192\n32","192\n64","256\n32","256\n64","512\n32","512\n64","1024\n32","1024\n64","2048\n32","2048\n64","4096\n32","4096\n64","8192\n32","8192\n64"]
y=[98.34,98.14,98.32,98.25,98.29,98.34,98.38,98.38,98.36,98.31,98.36,98.38,98.35,98.37,98.37,98.36,98.36,98.35,98.37,98.43,98.34]
ax.set_ylabel("Accuratezza", color='red')
ax.set_xlabel("Clusters")
ax.plot(x,y, '.-',label='first plot', color='red')
plt.grid()
#plt.ylim((93.5,98.5))
#plt.show()
#plt.title("Variazione dell’accuratezza della rete che ha appreso a classificare le cifre MNIST applicando la tecnica di weight sharing")
ax2 = ax.twinx()

x=["ALL"       ,"16\n8"   ,"32\n8"   ,"32\n16"  ,"64\n16"  ,"64\n32"  ,"128\n32","192\n32","192\n64","256\n32","256\n64","512\n32","512\n64","1024\n32","1024\n64","2048\n32","2048\n64","4096\n32","4096\n64","8192\n32","8192\n64"]
y=[None,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.51,0.51,0.52,0.52,0.53,0.53]
ax2.plot(x,y, '.-',label='first plot',color='blue')
#ax2.bar(x,y,label='first plot',color='blue', width=0.4,alpha=0.5)
ax2.set_ylabel(r'$r_2$', color='blue')
#ax2.grid()
plt.ylim((.48,.535))
#plt.grid()
plt.show()
#plt.savefig('png/ws_mnist.png')