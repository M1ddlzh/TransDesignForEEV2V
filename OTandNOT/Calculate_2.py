import math
import cvxpy as cp
import numpy as np

alpha1=0.5
alpha2=0.5
beta1=0.5
beta2=0.5
beta0=0
epsilon1=0.0001
cpln2 = 0.6931471805599453              # ln2 = 0.6931471805599453


def Pmin(rr1, rr2, delt, h1, h2, N):
    pmi1 = (2**(rr1/delt)-1)*N/h1
    pmi2 = (2**(rr2/(1-delt))-1)*N/h2
    return pmi1, pmi2

def TempP_EE(v, Q1, Q2, Z1, Z2, delt, eta, N, h11, h22, pmi1, pmi2, pma):
    pp1 = delt * ((2*Q1) / (math.log(2) * (2*Z1+v*alpha1)) - N / h11)
    pp2 = (1-delt) * ((2*Q2) / (math.log(2) * (2*Z2+v*alpha2)) - N / h22)
    # print(pp1, pp2)
    p11 = min(max(pp1, pmi1), pma)
    p22 = min(max(pp2, pmi2), pma)
    return p11, p22

def obj_EE(v, delt, h11, h22, p1, p2, N, eta, Z1, Z2, Q1, Q2, Y1, Y2, rr1, rr2, phi1, phi2):
    RR1 = delt*math.log2(1+h11*p1/N)
    RR2 = (1-delt)*math.log2(1+h22*p2/N)
    Tempobj = v*(alpha1*RR1+alpha2*RR2-eta*beta1*p1-beta2*eta*p2) \
        -2*(Z1*p1+Z2*p2+Q1*rr1+Q2*rr2-Y1*phi1-Y2*phi2)+2*(Q1*RR1+Q2*RR2)
    return Tempobj, RR1, RR2

def crosslayerCRNOT(r1, r2, h11, h12, h21, h22, N, Y1, Y2, Q1, Q2, Z1, Z2, V, pma, phi1, phi2):
    l = (2 ** r1 - 1) * (2 ** r2 - 1) * h12 * h21 / h11 / h22
    l1 = N * ((2 ** r1 - 1) * h21 / h11 + l) / (h21 * (1 - l))
    l2 = N * ((2 ** r2 - 1) * h12 / h22 + l) / (h12 * (1 - l))
    p_max = np.array([[pma], [pma]])    # 2 × 1

    if l < 1 and l1 <= pma and l2 <= pma:
        b1 = (2 ** r1 - 1) * N / h11
        b2 = (2 ** r2 - 1) * N / h22
        arr_B = np.array([[b1], [b2]])
        A = [h21*(2**r2-1)/h22,h12*(2**r1-1)/h11]
        if (1-A[0])*pma < arr_B[1]:
            pk2=pma
            pk1 = (pk2-arr_B[1])/A[0]
        elif (1-A[1])*pma < arr_B[0]:
            pk1=pma
            pk2=(pk1-arr_B[0])/A[1]
        else:
            pk1 = pma
            pk2 = pma

        a12 = h12 * (2 ** r1 - 1) / h11
        a21 = h21 * (2 ** r2 - 1) / h22
        arr_A = np.array([[-1, a12], [a21, -1]])
        I=0
        while True:
            I+=1
            p = cp.Variable(shape=(2, 1), nonneg=True)      # 2 × 1, np.dot(A, p) - B
            f = (V * alpha1 + 2 * Z1) * p[0][0] - V * beta1*phi1 - 2 * Y1*phi1 + 2 * Q1 * r1 + (V * alpha2 + 2 * Z2) * p[1][0] - V \
                * beta2*phi2 - 2 * Y2*phi2 + 2 * Q2 * r2 \
                - 2 * Q1 * cp.log(h11 * p[0][0] + h12 * p[1][0] + N) / cpln2 - 2 * Q2 * cp.log(
                h22 * p[1][0] + h21 * p[0][0] + N) / cpln2
            g = -2 * Q1 * np.log2(h12 * pk2 + N) - 2 * Q2 * np.log2(h21 * pk1 + N)
            deltaG = [-2 * Q2 * h21 / ((h21 * pk1 + N) * cp.log(2)),
                      -2 * Q1 * h12/ ((h12 * pk2 + N) * cp.log(2))]
            vectorP = [p[0][0] - pk1, p[1][0] - pk2]
            obje = cp.Minimize(f - g - np.dot(deltaG, vectorP))
            constr = [p <= p_max, arr_A @ p + arr_B <= 0]
            prob = cp.Problem(obje, constr)
            prob.solve(solver='SCS')
            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                if p.value[0] > 0 and p.value[1] > 0:
                    # print('optimal')
                    obj=obje.value
                    # print(obj)
                    pp1 = p.value[0]
                    pp2 = p.value[1]
                    Rk1 = math.log2(1 + h11 * pp1 / (N + h12 * pp2))
                    Rk2 = math.log2(1 + h22 * pp2 / (N + h21 * pp1))
                    if Rk1 < r1:
                        b1 = 0
                    else:
                        b1 = Rk1 - r1
                    if Rk2 < r2:
                        b2 = 0
                    else:
                        b2 = Rk2 - r2
                    # print('p1,pk1',p.value[0],pk1)
                    # print('p2,pk2', p.value[1], pk2)
                    if abs(p.value[0] - pk1) <= epsilon1 and abs(p.value[1] - pk2) <= epsilon1:
                        break
                    elif I>=3:
                        break
                    else:
                        pk1 = pp1
                        pk2 = pp2
                else:
                    pp1 = 0
                    pp2 = 0
                    obj = 1000
                    Rk1 = 0
                    Rk2 = 0
                    b1 = 0
                    b2 = 0
                    break
            else:
                pp1 = 0
                pp2 = 0
                obj= 1000
                Rk1 = 0
                Rk2 = 0
                b1=0
                b2=0
                break
    else:
        pp1 = 0
        pp2 = 0
        obj = 1000
        Rk1 = 0
        Rk2 = 0
        b1 = 0
        b2 = 0
    return obj, pp1, pp2, Rk1, Rk2, b1, b2
