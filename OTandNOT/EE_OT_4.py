import numpy as np
from tqdm import tqdm

from Calculate_4 import Pmin, TempP_EE, obj_EE


def avg_item(item):
    return sum(item) / len(item)


# np.random.seed(0)
phi = [[1, 1, 1, 1]]
delta = 1 / 4       # channel sharing coefficient
pmax_dbm = 100
pmin_dbm = 0
pmax = 100    
pavg = pmax / 4
pmin_w = 0    
T = 300
phi_th = 1
alpha1 = delta
alpha2 = delta
alpha3 = delta
alpha4 = delta
beta1 = delta
beta2 = delta
beta3 = delta
beta4 = delta

N0 = 1
scale_ii = np.sqrt(1 / 2)
band_width = 1
r = [0, 0.5]
lamda = 1
theta = lamda / band_width / 2

EE_OT = []  # energy efficiency
AvgsumPower_OT = []
Q_OT = []
REL_OT = []
AvgsumRate_OT = []

Q_max = 10 * lamda / band_width
large_than_q_OT = 0

for ep in range(33):
    for V in [0.1]: 
        Q1_OT = 0
        Q2_OT = 0
        Q3_OT = 0
        Q4_OT = 0
        Y1_OT = 0
        Y2_OT = 0
        Y3_OT = 0
        Y4_OT = 0
        Z1_OT = 0
        Z2_OT = 0
        Z3_OT = 0
        Z4_OT = 0
        eta_OT = 0.001
        avgR_OT = 0
        avgP_OT = 0
        sumQ_OT = 0.0
        rel_OT = 0

        for t in tqdm(range(T), desc=str(ep+1), ncols=60):
            h11 = (np.random.rayleigh(scale_ii)) ** 2
            h22 = (np.random.rayleigh(scale_ii)) ** 2
            h33 = (np.random.rayleigh(scale_ii)) ** 2
            h44 = (np.random.rayleigh(scale_ii)) ** 2

            a1 = np.random.poisson(lam = lamda) / band_width  # type-2
            a2 = np.random.poisson(lam = lamda) / band_width
            a3 = np.random.poisson(lam = lamda) / band_width
            a4 = np.random.poisson(lam = lamda) / band_width

            OBJ_OT = -10000
            R1_OT = 0
            R2_OT = 0
            R3_OT = 0
            R4_OT = 0
            P1_OT = 0
            P2_OT = 0
            P3_OT = 0
            P4_OT = 0
            PHI1_OT = 0
            PHI2_OT = 0
            PHI3_OT = 0
            PHI4_OT = 0
            b1_OT = 0
            b2_OT = 0
            b3_OT = 0
            b4_OT = 0
            for i in range(len(phi)):
                phi1 = phi[i][0]
                phi2 = phi[i][1]
                phi3 = phi[i][2]
                phi4 = phi[i][3]
                tempr1 = r[phi1]
                tempr2 = r[phi2]
                tempr3 = r[phi3]
                tempr4 = r[phi4]

                pmin1, pmin2, pmin3, pmin4 = Pmin(tempr1, tempr2, tempr3, tempr4, 
                    delta, h11, h22, h33, h44, N0)
                pmin1 = max(pmin1, pmin_w)
                pmin2 = max(pmin2, pmin_w)
                pmin3 = max(pmin3, pmin_w)
                pmin4 = max(pmin4, pmin_w)
                if pmin1 <= pmax and pmin2 <= pmax and pmin3 <= pmax and pmin4 <= pmax:
                    tp1, tp2, tp3, tp4 = TempP_EE(V, Q1_OT, Q2_OT, Q3_OT, Q4_OT, 
                        Z1_OT, Z2_OT, Z3_OT, Z4_OT, delta, eta_OT, N0, 
                        h11, h22, h33, h44, pmin1, pmin2, pmin3, pmin4, pmax)
                    tempobj, tempR1, tempR2, tempR3, tempR4 = obj_EE(V, delta, 
                        h11, h22, h33, h44, tp1, tp2, tp3, tp4, N0, eta_OT, 
                        Z1_OT, Z2_OT, Z3_OT, Z4_OT, Q1_OT, Q2_OT, Q3_OT, Q4_OT,
                        Y1_OT, Y2_OT, Y3_OT, Y4_OT, tempr1, tempr2, tempr3, tempr4, 
                        phi1, phi2, phi3, phi4)

                    if tempobj > OBJ_OT:    
                        OBJ_OT = tempobj
                        P1_OT = tp1
                        P2_OT = tp2    
                        P3_OT = tp3
                        P4_OT = tp4
                        R1_OT = min(tempR1, (Q1_OT + r[1]))
                        R2_OT = min(tempR2, (Q2_OT + r[1]))
                        R3_OT = min(tempR3, (Q3_OT + r[1]))
                        R4_OT = min(tempR4, (Q4_OT + r[1]))
                        if R1_OT > tempr1:
                            PHI1_OT = phi1
                            b1_OT = R1_OT - tempr1
                        else:
                            b1_OT = 0
                        if R2_OT > tempr2:
                            PHI2_OT = phi2
                            b2_OT = R2_OT - tempr2
                        else:
                            b2_OT = 0
                        if R3_OT > tempr3:
                            PHI3_OT = phi3
                            b3_OT = R3_OT - tempr3
                        else:
                            b3_OT = 0
                        if R4_OT > tempr4:
                            PHI4_OT = phi4
                            b4_OT = R4_OT - tempr4
                        else:
                            b4_OT = 0
                else:
                    P1_OT = 0
                    P2_OT = 0  
                    P3_OT = 0
                    P4_OT = 0
                    PHI1_OT = 0
                    PHI2_OT = 0
                    PHI3_OT = 0
                    PHI4_OT = 0
                    R1_OT = 0
                    R2_OT = 0
                    R3_OT = 0
                    R4_OT = 0
                    b1_OT = 0
                    b2_OT = 0
                    b3_OT = 0
                    b4_OT = 0

            Q1_OT = max(Q1_OT - b1_OT, 0) + a1
            Q2_OT = max(Q2_OT - b2_OT, 0) + a2
            Q3_OT = max(Q3_OT - b3_OT, 0) + a3
            Q4_OT = max(Q4_OT - b4_OT, 0) + a4
            Y1_OT = max(Y1_OT - PHI1_OT, 0) + phi_th
            Y2_OT = max(Y2_OT - PHI2_OT, 0) + phi_th
            Y3_OT = max(Y3_OT - PHI3_OT, 0) + phi_th
            Y4_OT = max(Y4_OT - PHI4_OT, 0) + phi_th
            if Q1_OT > Q_max or Q2_OT > Q_max or Q3_OT > Q_max or Q4_OT > Q_max:
                large_than_q_OT += 1
            avgR_OT += (alpha1 * R1_OT + alpha2 * R2_OT + alpha3 * R3_OT + alpha4 * R4_OT)
            avgP_OT += (beta1 * P1_OT + beta2 * P2_OT + beta3 * P3_OT + beta4 * P4_OT)
            sumQ_OT += (Q1_OT + Q2_OT + Q3_OT + Q4_OT)
            if avgP_OT == 0:
                eta_OT = 0
            else:
                eta_OT = avgR_OT / avgP_OT
            rel_OT += (PHI1_OT + PHI2_OT + PHI3_OT + PHI4_OT)
        if avgP_OT == 0:
            EE_OT.append(0)
        else:
            EE_OT.append(avgR_OT / avgP_OT)
        AvgsumPower_OT.append(avgP_OT / T)
        Q_OT.append(sumQ_OT / T / 4)
        REL_OT.append(rel_OT / T / 4)
        AvgsumRate_OT.append(avgR_OT / T)

print('OT-based')
print('EE', avg_item(EE_OT))
print('Avgsum_Power', avg_item(AvgsumPower_OT))
print('Q len', avg_item(Q_OT))
print('reli_EEOT', avg_item(REL_OT))
print('AvgsumRate', avg_item(AvgsumRate_OT))

print('large_than_qmax', large_than_q_OT)
