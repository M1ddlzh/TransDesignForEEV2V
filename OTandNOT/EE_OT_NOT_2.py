import numpy as np
from tqdm import tqdm

from Calculate_2 import Pmin, TempP_EE, obj_EE, crosslayerCRNOT


def avg_item(item):
    return sum(item) / len(item)


# np.random.seed(0)
phi = [[1, 1]]
delta = 0.5
pmax_dbm = 100
pmin_dbm = 0
pmax = 100
pavg = pmax / 4
pmin_w = 0
T = 300
phi_th = 1
alpha1 = 0.5
alpha2 = 0.5
beta1 = 0.5
beta2 = 0.5

N0 = 1
band_width = 1
r = [0, 0.5]
lamda = 1
theta = lamda / band_width / 2

scale_ii = np.sqrt(1 / 2)
scale_ij = np.sqrt(0.03 / 2)

EE_OT = []  # energy efficiency
AvgsumPower_OT = []
Q_OT = []

Q_CR = []
EE_CR = []
AvgsumPower_CR = []

fr_del = 0.5
R_FR = r[1] + lamda / band_width + theta
EE_FR = []
AvgsumPower_FR = []
Q_FR = []

REL_OT = []
REL_CR = []
REL_FR = []
AvgsumRate_FR = []
AvgsumRate_OT = []
AvgsumRate_CR = []

Q_max = 10 * lamda / band_width
large_than_q_OT = 0
large_than_q_CR = 0
large_than_q_FR = 0

##############################################################################################
for ep in range(33):
    for V in [1]:
        Q1_OT = 0
        Q2_OT = 0
        Y1_OT = 0
        Y2_OT = 0
        Z1_OT = 0
        Z2_OT = 0
        eta_OT = 0.001
        avgR_OT = 0
        avgP_OT = 0
        sumQ_OT = 0.0
        rel_OT = 0

        Q1_CR = 0
        Q2_CR = 0
        Y1_CR = 0
        Y2_CR = 0
        Z1_CR = 0
        Z2_CR = 0
        eta_CR = 0.001
        avgR_CR = 0
        avgP_CR = 0
        sumQ_CR = 0.0
        rel_CR = 0

        Q1_FR = 0
        Q2_FR = 0
        avgP_FR = 0
        sumQ_FR = 0.0
        avgR_FR = 0
        rel_FR = 0
        eta_FR = 0.001

        for t in tqdm(range(T), desc=str(ep+1), ncols=60):
            h11 = (np.random.rayleigh(scale_ii)) ** 2
            h22 = (np.random.rayleigh(scale_ii)) ** 2
            h12 = (np.random.rayleigh(scale_ij)) ** 2
            h21 = (np.random.rayleigh(scale_ij)) ** 2
            a1 = np.random.poisson(lam = lamda) / band_width  # type-2
            a2 = np.random.poisson(lam = lamda) / band_width

            OBJ_OT = -10000
            OBJ_CR = 10000
            OBJ_FR = 10000

            R1_OT = 0
            R2_OT = 0
            b2_OT = 0
            b1_OT = 0
            P1_OT = 0
            P2_OT = 0
            PHI1_OT = 0
            PHI2_OT = 0

            b1 = 0
            b2 = 0
            b1_CR = 0
            b2_CR = 0
            P1_CR = 0
            P2_CR = 0
            R1_CR = 0
            R2_CR = 0
            PHI1_CR = 0
            PHI2_CR = 0

            PHI2_FR = 0
            PHI1_FR = 0
            p1_FR = 0
            p2_FR = 0
            for i in range(len(phi)):
                phi1 = phi[i][0]
                phi2 = phi[i][1]
                tempr1 = r[phi1]
                tempr2 = r[phi2]

                ###########################################################################
                ###########################################################################
                ###########################################################################
                """
                EE OT
                """
                pmin1, pmin2 = Pmin(tempr1, tempr2, delta, h11, h22, N0)
                pmin1 = max(pmin1, pmin_w)
                pmin2 = max(pmin2, pmin_w)
                if pmin1 <= pmax and pmin2 <= pmax:
                    tp1, tp2 = TempP_EE(V, Q1_OT, Q2_OT, Z1_OT, Z2_OT, delta, eta_OT, N0, h11, h22, pmin1, pmin2, pmax)
                    tempobj, tempR1, tempR2 = obj_EE(V, delta, h11, h22, tp1, tp2, N0, eta_OT, Z1_OT, Z2_OT, Q1_OT, Q2_OT,
                                                    Y1_OT, Y2_OT,
                                                    tempr1, tempr2, phi1, phi2)
                    if tempobj > OBJ_OT:
                        OBJ_OT = tempobj
                        P1_OT = tp1
                        P2_OT = tp2
                        R1_OT = min(tempR1, (Q1_OT + r[1]))
                        R2_OT = min(tempR2, (Q1_OT + r[1]))
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
                else:
                    P1_OT = 0
                    P2_OT = 0
                    PHI1_OT = 0
                    PHI2_OT = 0
                    R1_OT = 0
                    R2_OT = 0
                    b1_OT = 0
                    b2_OT = 0

                ###########################################################################
                ###########################################################################
                ###########################################################################
                '''
                Cross-layer method
                '''
                tempobj, tp1, tp2, tempR1, tempR2, b1, b2 \
                = crosslayerCRNOT(tempr1, tempr2, h11, h12, h21, h22, N0, Y1_CR,
                                Y2_CR, Q1_CR, Q2_CR, Z1_CR, Z2_CR, V, pmax,
                                phi1, phi2)
                # print(tp1, tp2, tempR1, tempR2, b1, b2)

                if tempobj < OBJ_CR:
                    OBJ_CR = tempobj
                    P1_CR = tp1
                    P2_CR = tp2
                    R1_CR = min(tempR1, (Q1_CR + r[1]))
                    R2_CR = min(tempR2, (Q1_CR + r[1]))
                    if R1_CR > tempr1:
                        PHI1_CR = phi1
                    if R2_CR > tempr2:
                        PHI2_CR = phi2
                    b1_CR = b1
                    b2_CR = b2
                    POUT_CR = 0
                else:
                    R1_CR = 0
                    R2_CR = 0
                    P1_CR = 0
                    P2_CR = 0
                    b1_CR = 0
                    b2_CR = 0
                    PHI1_CR = 0
                    PHI2_CR = 0

            '''fixed-rate'''
            ###########################################################################
            ###########################################################################
            ###########################################################################
            ##OT
            tpfr11 = (2 ** (R_FR / fr_del) - 1) * N0 / h11
            tpfr22 = (2 ** (R_FR / (1 - fr_del)) - 1) * N0 / h22
            tpfr11 = max(tpfr11, pmin_w)
            tpfr22 = max(tpfr22, pmin_w)
            if tpfr11 >= 0 and tpfr11 <= pmax:
                tp11 = tpfr11
                phi1_fr = 1
            else:
                tp11 = pmax
                phi1_fr = 0
            if tpfr22 >= 0 and tpfr22 <= pmax:
                tp22 = tpfr22
                phi2_fr = 1
            else:
                tp22 = pmax
                phi2_fr = 0
            tempObjFr = beta1 * tp11 + beta2 * tp22
            if tempObjFr < OBJ_FR:
                p1_FR = tp11
                p2_FR = tp22
                OBJ_FR = tempObjFr
                PHI1_FR = phi1_fr
                PHI2_FR = phi2_fr

            ##NOT
            l = ((2 ** R_FR - 1) ** 2) * h12 * h21 / (h11 * h22)
            tpfr11 = (N0 * ((2 ** R_FR - 1) * h21 / h11 + l) / (h21 * (1 - l)))
            tpfr22 = (N0 * ((2 ** R_FR - 1) * h12 / h22 + l) / (h12 * (1 - l)))
            tpfr11 = max(tpfr11, pmin_w)
            tpfr22 = max(tpfr22, pmin_w)
            if tpfr11 >= 0 and tpfr11 <= pmax:
                tp11 = tpfr11
                phi1_fr = 1
            else:
                tp11 = pmax
                phi1_fr = 0
            if tpfr22 > 0 and tpfr22 <= pmax:
                tp22 = tpfr22
                phi2_fr = 1
            else:
                tp22 = pmax
                phi2_fr = 0
            tempObjFr = beta1 * tp11 + beta2 * tp22
            if tempObjFr < OBJ_FR:
                p1_FR = tp11
                p2_FR = tp22
                OBJ_FR = tempObjFr
                PHI1_FR = phi1_fr
                PHI2_FR = phi2_fr
            nf = [((2 ** R_FR) - 1) * N0 / h11, ((2 ** R_FR) - 1) * N0 / h22]
            k = ((2 ** R_FR - 1) ** 2) * h11 * h22 / h12 / h21
            pi = [N0 * ((2 ** R_FR - 1) * h11 / h21 + k) / (h11 * (1 - k)),
                N0 * ((2 ** R_FR - 1) * h22 / h12 + k) / (h22 * (1 - k))]
            pp = [((2 ** R_FR) * ((2 ** R_FR) - 1) * N0 / h12), ((2 ** R_FR) * (2 ** R_FR - 1) * N0 / h21)]
            if nf[0] <= pi[0] <= pmax and nf[1] <= pi[1] <= pmax:
                tpfr11 = pi[0]
                tpfr22 = pi[1]
            elif pi[0] <= nf[0] <= pmax and nf[1] <= pp[0] <= pmax:
                tpfr11 = nf[0]
                tpfr22 = pp[0]
            elif nf[0] <= pp[1] <= pmax and pi[1] <= nf[1] <= pmax:
                tpfr11 = pp[1]
                tpfr22 = nf[1]
            elif pi[0] <= nf[0] <= pmax and \
                (2 ** R_FR - 1) * N0 * (h11 * (2 ** R_FR - 1) / h22 + 1)\
                / h12 <= nf[1] <= N0 * (1 / h11 - 1 / h21) * h21 / h22:
                tpfr11 = nf[0]
                tpfr22 = nf[1]
            else:
                tpfr11 = pmax
                tpfr22 = pmax

            if tpfr11 >= 0 and tpfr11 <= pmax:
                tp11 = tpfr11
                phi1_fr = 1
            else:
                tp11 = pmax
                phi1_fr = 0
            if tpfr22 > 0 and tpfr22 <= pmax:
                tp22 = tpfr22
                phi2_fr = 1
            else:
                tp22 = pmax
                phi2_fr = 0
            tempObjFr = beta1 * tp11 + beta2 * tp22
            if tempObjFr < OBJ_FR:
                p1_FR = tp11
                p2_FR = tp22
                OBJ_FR = tempObjFr
                PHI1_FR = phi1_fr
                PHI2_FR = phi2_fr
            # SIC+Noise
            tpfr11 = nf[0]
            tpfr22 = nf[1] * h22 * max((2 ** R_FR) / h12, (h11 + h21 * (2 ** R_FR - 1)) / (h11 * h22))
            if tpfr11 >= 0 and tpfr11 <= pmax:
                tp11 = tpfr11
                phi1_fr = 1
            else:
                tp11 = pmax
                phi1_fr = 0
            if tpfr22 > 0 and tpfr22 <= pmax:
                tp22 = tpfr22
                phi2_fr = 1
            else:
                tp22 = pmax
                phi2_fr = 0
            tempObjFr = beta1 * tp11 + beta2 * tp22
            if tempObjFr < OBJ_FR:
                p1_FR = tp11
                p2_FR = tp22
                OBJ_FR = tempObjFr
                PHI1_FR = phi1_fr
                PHI2_FR = phi2_fr
            R1_FR = min(R_FR, r[1] + Q1_FR)
            R2_FR = min(R_FR, r[1] + Q2_FR)

            Q1_OT = max(Q1_OT - b1_OT, 0) + a1
            Q2_OT = max(Q2_OT - b2_OT, 0) + a2
            Y1_OT = max(Y1_OT - PHI1_OT, 0) + phi_th
            Y2_OT = max(Y2_OT - PHI2_OT, 0) + phi_th
            if Q1_OT > Q_max or Q2_OT > Q_max:
                large_than_q_OT += 1

            Q1_CR = max(Q1_CR - b1_CR, 0) + a1  # cross-layer
            Q2_CR = max(Q2_CR - b2_CR, 0) + a2
            Y1_CR = max(Y1_CR - PHI1_CR, 0) + phi_th
            Y2_CR = max(Y2_CR - PHI2_CR, 0) + phi_th
            Z1_CR = max(Z1_CR - pavg, 0) + P1_CR
            Z2_CR = max(Z2_CR - pavg, 0) + P2_CR
            if Q1_CR > Q_max or Q2_CR > Q_max:
                large_than_q_CR += 1

            Q1_FR = max(Q1_FR - (R_FR - r[PHI1_FR]), 0) + a1  # fixed-rate
            Q2_FR = max(Q2_FR - (R_FR - r[PHI2_FR]), 0) + a2 
            avgR_OT += (alpha1 * R1_OT + alpha2 * R2_OT)
            avgP_OT += (beta1 * P1_OT + beta2 * P2_OT)
            if Q1_FR > Q_max or Q2_FR > Q_max:
                large_than_q_FR += 1
            sumQ_OT += (Q1_OT + Q2_OT)
            if avgP_OT == 0:
                eta_OT = 0
            else:
                eta_OT = avgR_OT / avgP_OT

            avgR_CR += (alpha1 * R1_CR + alpha2 * R2_CR)
            avgP_CR += (beta1 * P1_CR + beta2 * P2_CR)
            sumQ_CR += (Q1_CR + Q2_CR)
            if avgP_CR == 0:
                eta_CR = 0
            else:
                eta_CR = avgR_CR / avgP_CR

            avgR_FR += (alpha1 * R1_FR + alpha2 * R2_FR)
            avgP_FR += (beta1 * p1_FR + beta2 * p2_FR)
            sumQ_FR += (Q1_FR + Q2_FR)
            if avgP_FR == 0:
                eta_FR = 0
            else:
                eta_FR = avgR_FR / avgP_FR
            rel_OT += (PHI1_OT + PHI2_OT)
            rel_CR += (PHI1_CR + PHI2_CR)
            rel_FR += (PHI1_FR + PHI2_FR)

        if avgP_OT == 0:
            EE_OT.append(0)
        else:
            EE_OT.append(avgR_OT / avgP_OT)
        AvgsumPower_OT.append(avgP_OT / T)
        Q_OT.append(sumQ_OT / T / 2)
        REL_OT.append(rel_OT / T / 2)
        AvgsumRate_OT.append(avgR_OT / T)

        if avgP_CR == 0:
            EE_CR.append(0)
        else:
            EE_CR.append(avgR_CR / avgP_CR)

        AvgsumPower_CR.append(avgP_CR / T)
        Q_CR.append(sumQ_CR / T / 2)
        REL_CR.append(rel_CR / T / 2)
        AvgsumRate_CR.append(avgR_CR / T)

        EE_FR.append(avgR_FR / avgP_FR)
        AvgsumPower_FR.append(avgP_FR / T)
        Q_FR.append(sumQ_FR / T / 2)
        REL_FR.append(rel_FR / T / 2)
        AvgsumRate_FR.append(avgR_FR / T)

print('-'*10 + 'OT-based')
print('EE', avg_item(EE_OT))
print('Avgsum_Power', avg_item(AvgsumPower_OT))
print('Q len', avg_item(Q_OT))
print('reli_EEOT', avg_item(REL_OT))
print('AvgsumRate', avg_item(AvgsumRate_OT))

print('-'*10 + 'PM-based')
print('crossE_E', avg_item(EE_CR))
print('Avgsum_Power', avg_item(AvgsumPower_CR))
print('Q len', avg_item(Q_CR))
print('reli_cr', avg_item(REL_CR))
print('AvgsumRate', avg_item(AvgsumRate_CR))

print('-'*10 + 'CSI-based')
print('fixed-fr_EE', avg_item(EE_FR))
print('fr_AvgsumPower', avg_item(AvgsumPower_FR))
print('fr_Q len', avg_item(Q_FR))
print('reli_FR', avg_item(REL_FR))
print('AvgsumRate', avg_item(AvgsumRate_FR))

print('large_than_qmax', large_than_q_OT, large_than_q_CR, large_than_q_FR)
