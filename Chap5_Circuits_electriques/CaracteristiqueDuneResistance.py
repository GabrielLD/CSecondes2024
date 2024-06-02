import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpltools import annotation

import ruler

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 12,
})
# Données expérimentales mesurées :

R = 200 # Ohm
I = np.array([0,10,20,30,40,50,60]) # mA
I = I*1e-3
U = R*I

# détermination des coefficients
m_I_U = np.polyfit(I,U,1)         # le dernier nombre donne le degré du polynôme
#Affichage des résultats
print('Modélisation de x=f(t) par une fonction affine : x = at+b')
print('  a =',m_I_U[0])
print('  b =',m_I_U[1])
Imodel = np.linspace(0,60e-3,10000)
Umodel = Imodel *m_I_U[0]



import matplotlib.pyplot as plt



#
# plt.figure(figsize=(8.27/2, 11.69/4), dpi=100)
# plt.plot(I,U,'o', color = 'steelblue', markersize = 7, label = r'Mesures expérimentales')
# plt.plot(Imodel, Umodel, color = 'darkcyan', label = r'$U = R\times I$')
# plt.annotate(xy=(0.012,6), text=r'$R = 200 \Omega$')
# # annotation.slope_marker((0.04, 6), 200)
# plt.xlabel(r'Intensité $I$ (A)')
# plt.ylabel(r'Tension $U$ (V)')
# plt.legend(fontsize = 8)
# plt.grid()
# plt.tight_layout()
# plt.savefig('CaractéristiqueCourantTension.png', dpi = 300)
# plt.show()



r = 50 # Ohm
E = 9 # V
IG = np.linspace(0,60e-3,10) # mA
UG = E-r*IG

# détermination des coefficients
m_I_U = np.polyfit(IG,UG,1)         # le dernier nombre donne le degré du polynôme
#Affichage des résultats
print('Modélisation de x=f(t) par une fonction affine : x = at+b')
print('  a =',m_I_U[0])
print('  b =',m_I_U[1])
IGmodel = np.linspace(0,60e-3,10000)
UGmodel = Imodel *m_I_U[0] + m_I_U[1]

#
# plt.figure(figsize=(8.27/2, 11.69/4), dpi=100)

# plt.plot(IG,UG,'o', color = 'darkcyan', markersize = 7, label = r'Mesures expérimentales')
# plt.plot(IGmodel, UGmodel, color = 'springgreen', label = r'$U = E-r\times I$')
# plt.xlabel(r'Intensité $I$ (A)')
# plt.ylabel(r'Tension $U$ (V)')
# plt.legend(fontsize = 8)
# plt.axis([0, 0.07, 0 , 10])
# plt.grid()
# plt.tight_layout()
# plt.savefig('CaractéristiqueGenerateur.png', dpi = 300)
# plt.show()

plt.figure(figsize=(8.27/2, 11.69/4), dpi=100)
plt.plot(I,U,'o', color = 'darkcyan', markersize = 5)
plt.plot(IG,UG,'o', color = 'springgreen', markersize = 5)
plt.plot(IGmodel, UGmodel, color = 'springgreen', linewidth=1.8, label = r'Caractéristique de la pile')
plt.plot(Imodel, Umodel, color = 'darkcyan', linewidth=1.8, label = r'Caractéristique de la résistance')
plt.xlabel(r'Intensité $I$ (A)')
plt.ylabel(r'Tension $U$ (V)')
plt.legend(fontsize = 8)
# plt.annotate(xy=(0.0360,7), xytext=(0.037,3), text=r'Point de',arrowprops=dict(facecolor='black', shrink=0.1), fontsize = 10)
# plt.annotate(xy=(0.0360,7), xytext=(0.0370,2.2), text=r'fonctionnement', fontsize = 10)
# plt.axis([0, 1.2, 0 , 6])
# plt.vlines(x = 0.0360,ymin = -0.8, ymax = 7, color = 'red')
# plt.annotate(xy=(0.0370,1.), text= r'$P(I_f,U_f)$')
# plt.hlines(y = 7,xmin = -0.01, xmax = 0.0360, color = 'red')
# plt.annotate(xy=(-0.01,7), text= r'$U_f$', color = 'red')

plt.grid()
plt.tight_layout()
plt.savefig('PointDeFonctionnement2.png', dpi = 300)
plt.show()
#

# Ilampe = np.array([0,0.018,0.035,0.051,0.058,0.066,0.080, 0.089, 0.095,0.103,0.108,0.115,0.119,0.124,0.129])
# Ulampe = np.array([0,0.25,0.88,1.72,2.16,2.72,3.76,4.72,5.28,6.08,6.64,7.48,7.88,8.51,9.20])
#
#
# plt.figure(figsize=(8.27/2, 11.69/4), dpi=100)
# plt.plot(Ilampe,Ulampe, 'o-', color = 'darkcyan', linewidth=1.8, label = r'Caractéristique de la lampe')
# plt.xlabel(r'Intensité $I$ (A)')
# plt.ylabel(r'Tension $U$ (V)')
# plt.grid()
# plt.tight_layout()
# plt.savefig('CaracteristiqueLampe.png', dpi = 300)
# plt.show()