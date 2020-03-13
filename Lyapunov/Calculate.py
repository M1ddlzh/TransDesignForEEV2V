import cvxpy as cp
import numpy as np

"""
Windows 10: cvxpy 1.0.21 and Ubuntu 16.04: cvxpy 1.0.27 have been tested.
"""

alpha1 = 0.5
alpha2 = 0.5
beta1 = 0.5
beta2 = 0.5

cpln2 = 0.6931471805599453              # ln2 = 0.6931471805599453

def NOA(h11, h12, h21, h22, v, QQ1, QQ2, r1, r2, N, phii1, phii2, pma, eta, YY1, YY2):
    epsilon = 0.01                      # iteration breaking threshold

    k = (2 ** r1 - 1) * (2 ** r2 - 1) * h12 * h21 / (h11 * h22)     
    pnot1 = N * ((2 ** r1 - 1) * h21 / h11 + k) / (h21 * (1 - k))       # minimum power required
    pnot2 = N * ((2 ** r2 - 1) * h12 / h22 + k) / (h12 * (1 - k))
    p_max = np.array([[pma], [pma]])    # 2 × 1
    p_min = np.array([[0], [0]])        # 2 × 1

    if k < 1 and pnot1 <= pma and pnot2 <= pma:
        I = 0                           # iteration num
        objNOA = -10000

        # initialize pk1, pk2
        if r1 > 0:      
            pt1 = h11 * pma / (h12 * (2 ** r1 - 1)) - N / h12
        else:
            pt1 = pma

        if r2 > 0:
            pt2 = h22 * pma / (h21 * (2 ** r2 - 1)) - N / h21
        else:
            pt2 = pma

        if 0 < pt1 < pma:
            pk1 = pma
            pk2 = pt1
        elif 0 < pt2 < pma:
            pk1 = pt2
            pk2 = pma
        else:
            pk1 = pma
            pk2 = pma

        while True:
            I += 1
            p = cp.Variable(shape=(2, 1), nonneg=True)                                          # 2 × 1, np.dot(A, p) - B
            # if use "f = (...) / cp.log(2)", will raise error "Problem does not follow DCP rules."
            f = (v * alpha1 + 2 * QQ1) * cp.log(N + h11 * p[0][0] + h12 * p[1][0]) / cpln2 + \
                (v * alpha2 + 2 * QQ2) * cp.log(N + h22 * p[1][0] + h21 * p[0][0]) / cpln2 - \
                (v * eta * beta1) * p[0][0] - (v * eta * beta2) * p[1][0] - \
                2 * (QQ1 * r1 - phii1 * YY1 + QQ2 * r2 - phii2 * YY2)                           # u_i, v_i = 1, 1, omit
            y1 = N + h12 * pk2
            y2 = N + h21 * pk1
            g = (v * alpha1 + 2 * QQ1) * cp.log(y1) / cpln2 + (v * alpha2 + 2 * QQ2) * cp.log(y2) / cpln2
            
            vectorP = np.array([p[0][0] - pk1, p[1][0] - pk2])
            deltaG = np.array([(v*alpha2 + 2 * QQ2) * h21 / ((N + h21 * pk1) * cpln2),
                               (v*alpha1 + 2 * QQ1) * h12 / ((N + h12 * pk2) * cpln2)])

            A = np.array([[-1, h12 * (2 ** r1 - 1) / h11], [h21 * (2 ** r2 - 1) / h22, -1]])    # 2 × 2
            B = np.array([[-(2 ** r1 - 1) * N / h11], [-(2 ** r2 - 1) * N / h22]])              # 2 × 1                   
           
            objfunc = cp.Maximize(f - g - deltaG[0] * vectorP[0] - deltaG[1] * vectorP[1])  
            constr = [p_min <= p, p <= p_max, (A * p - B) <= 0 ]
            prob = cp.Problem(objfunc, constr)
            # print(prob)
            prob.solve(solver = cp.SCS)
            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                pp1 = max(p.value[0][0], 0)
                pp2 = max(p.value[1][0], 0)
                k1 = h11 * pp1 / (N + h12 * pp2)        # SINR
                k2 = h22 * pp2 / (N + h21 * pp1)
                if k1 >= 0 and k2 >= 0:
                    Rk1 = np.log2(1 + k1)               # channel capacity, bps/Hz
                    Rk2 = np.log2(1 + k2)
                else:
                    pp1 = 0
                    pp2 = 0
                    objNOA = prob.value
                    Rk1 = 0
                    Rk2 = 0
                if abs(objNOA - objfunc.value) <= epsilon:
                    objNOA = objfunc.value
                    break
                elif I >= 20:
                    objNOA = objfunc.value
                    break
                else:
                    pk1 = pp1
                    pk2 = pp2
                    objNOA = objfunc.value
            else:
                pp1 = 0
                pp2 = 0
                objNOA = -10000
                Rk1 = 0
                Rk2 = 0
                break
    else:
        pp1 = 0
        pp2 = 0
        objNOA= -10000
        Rk1 = 0
        Rk2 = 0
    return objNOA, pp1, pp2, Rk1, Rk2

