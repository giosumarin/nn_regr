import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
x=["NN3",1  ,2  ,3  ,4  ,5  ,6  ,7  ,8  ,9  ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,26 ,30 ,34 ,38 ,42 ,46 ,50 ,54 ,58 ,62 ,72 ,80 ,88 ,96 ,104,112,120,128]
y=[1270,714,593,494,515,545,561,538,465,453,425,338,311,207,238,254,305,266,246,193,233,277,357,209,233,196,177,214,153,143,145,133,143,121,134,153,109,107,99 ,118,302]
ax.set_ylabel(r'$\epsilon$', color='red')
ax.set_xlabel("Numero di Split")
p1=ax.plot(x,y, '-',label='first plot', color='red')
plt.grid()
plt.xticks(rotation=70)
#plt.ylim((95,1300))
#plt.show()
y=[None, 714.0  ,557.5  ,420.33 ,428.75 ,360.0  ,319.17 ,292.57 ,298.5  ,240.33 ,228.6  ,222.64 ,200.08 ,183.85 ,175.07 ,176.53 ,196.75 ,169.53 ,154.06 ,147.79 ,144.3  ,150.67 ,143.0  ,131.85 ,116.7  ,106.15 ,104.55 ,101.38 ,88.57  ,84.88  ,85.07  ,82.4   ,80.34  ,69.71  ,66.75  ,65.88  ,60.6   ,59.13  ,56.11  ,54.53  ,56.22] 
p2=ax.plot(x,y, '-',label='first plot', color='orange')
#plt.title("Variazione dell’accuratezza della rete che ha appreso a classificare le cifre MNIST applicando la tecnica di weight sharing")
ax2 = ax.twinx()
ax.legend((p1[0], p2[0]), ("Massimo","Media"), loc='upper center')

x=["NN3",1  ,2  ,3  ,4  ,5  ,6  ,7  ,8  ,9  ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,26 ,30 ,34 ,38 ,42 ,46 ,50 ,54 ,58 ,62 ,72 ,80 ,88 ,96 ,104,112,120,128]
#plt.ylim((.48,.535))
y=[None,0.000786,0.001571,0.002357,0.003146,0.003929,0.004708,0.005487,0.006299,0.007078,0.007857,0.008636,0.009448,0.010227,0.011006,0.011786,0.012565,0.013377,0.014156,0.014935,0.015714,0.016494,0.017305,0.020455,0.023571,0.026721,0.02987,0.033117,0.036039,0.039286,0.042532,0.045455,0.048701,0.056494,0.062987,0.069156,0.075325,0.081818,0.087987,0.094481,0.100649]
plt.xticks(rotation=70)
ax2.plot(x,y, '-',label='first plot',color='blue')
#ax2.bar(x,y,label='first plot',color='blue', width=0.4,alpha=0.5)
ax2.set_ylabel(r'$r_3$', color='blue')
#ax2.grid()
#plt.ylim((.48,.535))
#plt.grid()

plt.show()
#plt.savefig('png/nets.svg', format='svg')
