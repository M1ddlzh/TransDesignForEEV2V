import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Calculate import Pmin, TempP, TempP_EE, obj, obj_EE, NOA, crosslayer


np.random.seed(0)

global R1, R2, R_1, R_2, Rvv_1, fr_r1, fr_r2

phi = [[1, 1], [1, 0], [0, 1], [0, 0]]  # 4 actions

phi_th = 0.9        # ty1 transmission success shreshold

alpha1 = 0.5        # weight of R and P of two pairs of vehicles
alpha2 = 0.5
beta1 = 0.5
beta2 = 0.5

dii = 10            # distance, m
d21 = 100
d12 = 120
N0 = 10**(-15)          # W/Hz
band_width = 1e5        # 100 kHz
N0 = N0 * band_width    # Watt
PLii = - (103.4 + 24.2 * np.log10(dii / 1000))
PL12 = - (103.4 + 24.2 * np.log10(d21 / 1000))
PL21 = - (103.4 + 24.2 * np.log10(d12 / 1000))
sigmaii = (10**(PLii / 10))
sigma12 = (10**(PL12 / 10))    
sigma21 = (10**(PL21 / 10))      

T = 1000                    # time slot

VV = np.arange(2, 7, 2)     # weight of energy efficiency

r = [0, 200]                # ty1 transmission rate
lamda = 500                 # ty2 generate rate
pmax_dbm = np.arange(23, 24, 3)     # power, dBm
pnum = len(pmax_dbm)

EE = [[],[],[]]             #energy efficiency
AvgsumPower = [[],[],[]]
AvgsumRate = [[], [], []]
Y = [[],[],[]]              # ty1 queue
Q = [[],[],[]]              # ty2 queue
REL = [[], [], []]          # reliability of ty1

for vnum in range(len(VV)):
    V = VV[vnum]
    for a in range(pnum):
        P_max = pmax_dbm[a]
        pmax = (10 ** (P_max / 10)) / 1000      # dBm -> W
        Y1 = 0.0
        Y2 = 0.0
        Q1 = 0.0 
        Q2 = 0.0
        eta = 0.0       # ee
        Rv1 = 0.0       # sum rate
        Pv1 = 0.0       # sum power
        sumQ = 0.0
        sumY = 0.0
        rel = 0.0
        for t in tqdm(range(T), ncols = 60):
            h11 = np.random.gamma(2, sigmaii / 2)  
            h22 = np.random.gamma(2, sigmaii / 2)
            h12 = np.random.gamma(2, sigma12 / 2)
            h21 = np.random.gamma(2, sigma21 / 2)
            # print(h11, h22, h12, h21)

            a1 = np.random.poisson(lam = lamda)     # ty2 data
            a2 = np.random.poisson(lam = lamda)

            OBJ = -1000         # a small initial objective
            R1 = 0
            R2 = 0
            P1 = 0
            P2 = 0
            b1_ee = 0           # ty2 transmission rate
            b2_ee = 0
            PHI1 = 0            # the best of the four actions
            PHI2 = 0

            for i in range(4):  
                phi1 = phi[i][0]    # transfer ty1 or not
                phi2 = phi[i][1]
                tempr1 = r[phi1]
                tempr2 = r[phi2]

                tempobj, tp1, tp2, tempR1, tempR2 = \
                    NOA(h11, h12, h21, h22, V, Q1, Q2, tempr1, tempr2, N0, phi1, phi2, pmax, eta, Y1, Y2)

                if tempobj > OBJ:
                    OBJ = tempobj
                    P1 = tp1
                    P2 = tp2
                    PHI1 = phi1
                    PHI2 = phi2
                    R1 = tempR1
                    R2 = tempR2
                    if R1 > tempr1:
                        b1_ee = R1 - tempr1
                    else:
                        b1_ee = 0
                    if R2 > tempr2:
                        b2_ee = R2 - tempr2
                    else:
                        b2_ee = 0
            
            sumQ += (Q1 + Q2)
            sumY += (Y1 + Y2)
            Y1 = max(Y1 - PHI1, 0) + phi_th     # ty1 queue
            Y2 = max(Y2 - PHI2, 0) + phi_th
            Q1 = max(Q1 - b1_ee, 0) + a1        # ty2 queue
            Q2 = max(Q2 - b2_ee, 0) + a2
            
            Rv1 += (alpha1 * R1 + alpha2 * R2)
            Pv1 += (beta1 * P1 + beta2 * P2)

            if Pv1 == 0.0:
                eta = 0.0
            else:
                eta = Rv1 / Pv1
            rel += (PHI1 + PHI2)    

        if Pv1 == 0.0:
            EE[vnum].append(0.0)
        else:
            EE[vnum].append(round((Rv1 / Pv1), 3))

        AvgsumPower[vnum].append(((Pv1) / T))
        Q[vnum].append(round((sumQ) / T, 3))
        REL[vnum].append(round(rel / (2 * T), 3))
        AvgsumRate[vnum].append(round(Rv1 / T, 3))
        Y[vnum].append(round((sumY) / T, 3))

path = './fig_result/'
if not os.path.exists(path):
    os.makedirs(path)

fig1 = plt.figure("0420 23 36Achievable energy efficiency") 
PE1 = plt.semilogy(pmax_dbm, EE[0], color='blue', marker='o', label='EE-based design, V=' + str(VV[0]))
PE2 = plt.semilogy(pmax_dbm, EE[1], color='orange', marker='*', label='EE-based design, V=' + str(VV[1]))
PE3 = plt.semilogy(pmax_dbm, EE[2], color='red', marker='o', label='EE-based design, V=' + str(VV[2]))
plt.xlabel("pmax_dbm")
plt.ylabel("Energy Efficiency")
plt.legend(loc='upper right')
plt.savefig(path + "Energy_Efficiency_" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

fig2 = plt.figure("Reliability of type-1 messages")
PR1 = plt.plot(pmax_dbm, REL[0], color='blue', linestyle='-', label='EE-based design, V=' + str(VV[0]))
PR2 = plt.plot(pmax_dbm, REL[1], color='orange', linestyle='-.', label='EE-based design, V=' + str(VV[1]))
PR3 = plt.plot(pmax_dbm, REL[2], color='red', linestyle='-', label='EE-based design, V=' + str(VV[2]))
plt.xlabel("pmax_dbm")
plt.ylabel("Reliability")
plt.legend(loc='upper right')
plt.savefig(path + "Reliability_" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

fig3= plt.figure("queue length of type-2 messages")
PQ1 = plt.plot(pmax_dbm, Q[0], color='blue', linestyle='-.', label='EE-based design, V=' + str(VV[0]))
PQ2 = plt.plot(pmax_dbm, Q[1], color='orange', linestyle='-.', label='EE-based design, pmax=' + str(VV[1]))
PQ3 = plt.plot(pmax_dbm, Q[2], color='red', linestyle='-.', label='EE-based design, pmax=' + str(VV[2]))
plt.xlabel("pmax_dbm")
plt.ylabel("Queue length")
plt.legend(loc='upper right')
plt.savefig(path + "Queue_length_" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

fig4 = plt.figure("AvgsumRate")
PRA1 = plt.plot(pmax_dbm, AvgsumRate[0], color='blue', linestyle='-', label='EE-based design,V=' + str(VV[0]))
PRA2 = plt.plot(pmax_dbm, AvgsumRate[1], color='orange', linestyle='-', label='EE-based design,V=' + str(VV[1]))
PRA3 = plt.plot(pmax_dbm, AvgsumRate[2], color='red', linestyle='-', label='EE-based design,V=' + str(VV[2]))
plt.xlabel("pmax_dbm")
plt.ylabel("AvgsumRate")
plt.legend(loc='upper right')
plt.savefig(path + "AvgsumRate_" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

fig5 = plt.figure("AvgsumPower")
VP1=plt.plot(pmax_dbm, AvgsumPower[0], color='blue', linestyle='--', label='EE-based design,V=' + str(VV[0]))
VP2=plt.plot(pmax_dbm, AvgsumPower[1], color='orange', linestyle='--', label='EE-based design,V=' + str(VV[1]))
VP3=plt.plot(pmax_dbm, AvgsumPower[2], color='red', linestyle='-', label='EE-based design,V=' + str(VV[2]))
plt.xlabel('pmax_dbm')
plt.ylabel('AvgsumPower')
plt.legend(loc='upper right')
plt.savefig(path + "AvgsumPower_" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

plt.show()


