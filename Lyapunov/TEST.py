r"""
Original Paper: D. Lan, C. Wang, P. Wang, F. Liu and G. Min, "Transmission Design for Energy-Efficient 
Vehicular Networks with Multiple Delay-Limited Applications," 2019 IEEE Global Communications Conference (GLOBECOM), 
Waikoloa, HI, USA, 2019, pp. 1-6.
doi: 10.1109/GLOBECOM38437.2019.9014246
keywords: {Optimization;Sensors;Reliability;Delays;Power demand;Encoding;Linear programming},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9014246&isnumber=9013108
"""


import os
import datetime
import numpy as np
from tqdm import tqdm
from Calculate import NOA


np.random.seed(0)

actions = [[1, 1], [1, 0], [0, 1], [0, 0]]

phi_th = 0.9                            # ty1 transmission success shreshold

alpha1 = 0.5                            # weight of R and P of two pairs of vehicles
alpha2 = 0.5
beta1 = 0.5
beta2 = 0.5

dii = 10                                # distance, metre
d21 = 100
d12 = 120
N0 = 10**(-15)                          # W/Hz
band_width = 100                        # 100 kHz
PLii = - (103.4 + 24.2 * np.log10(dii / 1000))
PL12 = - (103.4 + 24.2 * np.log10(d21 / 1000))
PL21 = - (103.4 + 24.2 * np.log10(d12 / 1000))
sigmaii = (10**(PLii / 10))
sigma12 = (10**(PL12 / 10))    
sigma21 = (10**(PL21 / 10))      

T = 1000                                # time slot

ALL_V = np.arange(20, 24, 5)            # weight of energy efficiency

r = [0, 100 / band_width]               # ty1 transmission rate, bit/Hz
lamda = 300 / band_width                # ty2 generate rate, bit/Hz

pmax_dbm = 23                           # power, dBm
pmax = (10 ** (pmax_dbm / 10)) / 1000   # dBm -> W

EE = []                                 # energy efficiency
AvgsumPower = []
AvgsumRate = []
Y = []                                  # ty1 queue
Q = []                                  # ty2 queue
RELI = []                               # reliability of ty1

for vnum in range(len(ALL_V)):
    V = ALL_V[vnum]                     
    Y1 = 0.0
    Y2 = 0.0
    Q1 = 0.0 
    Q2 = 0.0
    eta = 0.0                           # ee
    Rv = 0.0                            # sum rate
    Pv = 0.0                            # sum power
    sumQ = 0.0
    sumY = 0.0
    reli = 0.0
    for t in tqdm(range(T), ncols = 60):
        h11 = np.random.gamma(2, sigmaii / 2)  
        h22 = np.random.gamma(2, sigmaii / 2)
        h12 = np.random.gamma(2, sigma12 / 2)
        h21 = np.random.gamma(2, sigma21 / 2)

        a1 = np.random.poisson(lam = lamda)     # ty2 data
        a2 = np.random.poisson(lam = lamda)

        OBJ = -10000                            # a small initial objective
        R1 = 0
        R2 = 0
        P1 = 0
        P2 = 0
        b1 = 0                                  # ty2 transmission rate
        b2 = 0
        PHI1 = 0                                # the best of the four actions
        PHI2 = 0

        for i in range(len(actions)):  
            phi1 = actions[i][0]                # transfer ty1 or not
            phi2 = actions[i][1]
            tempr1 = r[phi1]
            tempr2 = r[phi2]

            tempobj, temp1, temp2, tempR1, tempR2 = \
                NOA(h11, h12, h21, h22, V, Q1, Q2, tempr1, tempr2, N0, phi1, phi2, pmax, eta, Y1, Y2)

            if tempobj > OBJ:
                OBJ = tempobj
                P1 = temp1
                P2 = temp2
                PHI1 = phi1
                PHI2 = phi2
                R1 = tempR1
                R2 = tempR2
                if R1 > tempr1:
                    b1 = R1 - tempr1
                else:
                    b1 = 0
                if R2 > tempr2:
                    b2 = R2 - tempr2
                else:
                    b2 = 0
        
        sumQ += (Q1 + Q2)
        sumY += (Y1 + Y2)
        Y1 = max(Y1 - PHI1, 0) + phi_th     # ty1 queue
        Y2 = max(Y2 - PHI2, 0) + phi_th
        Q1 = max(Q1 - b1, 0) + a1           # ty2 queue
        Q2 = max(Q2 - b2, 0) + a2
        
        Rv += (alpha1 * R1 + alpha2 * R2)
        Pv += (beta1 * P1 + beta2 * P2)

        if Pv == 0.0:
            eta = 0.0
        else:
            eta = Rv / Pv
        reli += (PHI1 + PHI2)  

    if Pv == 0.0:
        EE.append(0.0)
    else:
        EE.append(round(Rv / Pv, 4))
    AvgsumPower.append(round(Pv / T, 4))
    AvgsumRate.append(round(Rv / T, 4))
    Y.append(round(sumY / T, 4))
    Q.append(round(sumQ / T, 4))
    RELI.append(round(reli / (2 * T), 4))  

print('EE: ', EE, '\n', 'Y: ', Y, '\n', 'Q: ', Q, '\n', 'RELI: ', RELI, sep='')

path = './Lyapunov_results/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt', 'w+') as f:
    f.write('ALL_V: ' + str(ALL_V) + '\n')
    f.write('Energy Efficiency:'); f.write(str(EE) + '\n')
    f.write('Y:'); f.write(str(Y) + '\n')
    f.write('Q:'); f.write(str(Q) + '\n')
    f.write('RELI:'); f.write(str(RELI) + '\n')
    f.write('AvgsumPower:'); f.write(str(AvgsumPower) + '\n')
    f.write('AvgsumRate:'); f.write(str(AvgsumRate) + '\n')

