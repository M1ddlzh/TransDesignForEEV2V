import math
import cvxpy as cp
import numpy as np
# from sympy.abc import x,y
# import random


alpha1=0.5
alpha2=0.5
beta1=0.5
beta2=0.5
beta0=0
epsilon = 0.05
epsilon1=0.002
beta10=1
n=2

def Pmin(rr1, rr2, delt, h1, h2, N):
    pmi1 = (2**(rr1/delt)-1)*delt*N/h1
    pmi2 = (2**(rr2/(1-delt))-1)*(1-delt)*N/h2
    return pmi1,pmi2


def TempP_EE(v, Q1, Q2, Z1, Z2, delt, eta, N, h11, h22, pmi1, pmi2, pma):
    pp1 = (v*alpha1+2*Q1)*delt/(math.log(2)*(2*Z1+v*eta*beta1))-delt*N/h11   
    pp2 = (1-delt)*((v*alpha2+2*Q2)/((2*Z2+v*eta*beta2)*math.log(2))-N/h22)
    p11 = min(max(pp1, pmi1), pma)
    p22 = min(max(pp2, pmi2), pma)
    return p11, p22

def obj_EE(v, delt, h11, h22, p1, p2, N, eta, Z1, Z2, Q1, Q2, Y1, Y2, rr1, rr2, phi1, phi2):
    RR1 = delt*math.log2(1+h11*p1/(delt*N))
    RR2 = (1-delt)*math.log2(1+h22*p2/((1-delt)*N))
    Tempobj = v*(alpha1*RR1+alpha2*RR2-eta*beta1*p1-beta2*eta*p2)-2*(Z1*p1+Z2*p2+Q1*rr1+Q2*rr2-Y1*phi1-Y2*phi2)+2*(Q1*RR1+Q2*RR2)
    return Tempobj,RR1,RR2

    #####below cross-layer##
def TempP(v, Q1, Q2, Z1, Z2, delt, N, h11, h22, pmi1, pmi2, pma):
    pp1=delt*(2*Q1/((v*alpha1+2*Z1)*math.log(2))-N/h11)
    pp2=(1-delt)*(2*Q2/((v*alpha2+2*Z2)*math.log(2))-N/h22)
    p11 = min(max(pp1,pmi1), pma)
    p22 = min(max(pp2,pmi2), pma)
    return p11,p22

def obj(v, delt, h11, h22, p1, p2, N, Z1, Z2, Q1, Q2, Y1, Y2, rr1, rr2, phi1, phi2):
    RR1 = delt*math.log2(1+h11*p1/(delt*N))
    RR2 = (1-delt)*math.log2(1+h22*p2/((1-delt)*N))
    if RR1<rr1:
        b1=0
    else:b1=RR1-rr1
    if RR2<rr2:
        b2=0
    else:b2=RR2-rr2
    Tempobj = v * (alpha1 * p1 + alpha2 * p2 - beta1 * phi1 - beta2 * phi2) \
              -2 * (Y1*phi1 +Y2*phi2- Z1 * p1 - Z2 * p2)-2*Q1*b1-2*Q2*b2
    return Tempobj,RR1,RR2,b1,b2

def crosslayer(r1, r2, h11, h12, h21, h22, N, Y1, Y2, Q1, Q2, Z1, Z2, V, pma,phi1,phi2):
    # print('cr')
    epsilon1=0.002
    lcr = (2 ** r1 - 1) * (2 ** r2 - 1) * h12 * h21 / h11 / h22
    lcr1 = N * (((2 ** r1) - 1) * h21 / h11 + lcr) / (h21 * (1 - lcr))
    lcr2 = N * (((2 ** r2) - 1) * h12 / h22 + lcr) / (h12 * (1 - lcr))
    n=2
    p_max=pma*np.ones(n)
    p_min=np.zeros(n)
    # objcr = cp.Parameter()
    # print('pnot1,pnot2',lcr1,lcr2)
    if lcr<1 and lcr1 <= pma and lcr2 <= pma:

        objcr = 10000
        b1 = (2 ** r1 - 1) * N / h11
        b2 = (2 ** r2 - 1) * N / h22
        B = [-b1, -b2]


        if r1>0:
            pt1 = h11 * pma / (h12 * (2 ** r1 - 1)) - N / h12
        else:pt1=pma
        if r2>0:
            pt2 = h22 * pma / (h21 * (2 ** r2 - 1)) - N / h21
        else:pt2=pma
        if 0<pt1 and pt1 < pma:
            pk1 = pma
            pk2 = pt1
        elif 0<pt2 and pt2< pma:
            pk1 = pt2
            pk2 = pma
        else:
            pk1 = pma
            pk2 = pma

        I = 0
        while True:  
            I+=1
            a12 = h12 * ((2 ** r1) - 1) / h11
            a21 = h21 * ((2 ** r2) - 1) / h22
            A = [[-1, a12], [a21, -1]]

            p = cp.Variable(shape=(2,), nonneg=True)
            f = (V* beta1 + 2 * Z1) * p[0] - V * beta10*phi1 - 2 * Y1*phi1 + 2 * Q1 * r1 + (V * beta2+ 2 * Z2) * p[1] - V \
                * beta10*phi2 - 2 * Y2*phi2 + 2 * Q2 * r2 \
                - 2 * Q1 * np.log2(math.e) * cp.log(h11 * p[0] + h12 * p[1] + N) - 2 * Q2 * np.log2(math.e) * cp.log(
                h22 * p[1] + h21 * p[0] + N)

            g = -2 * Q1 *np.log2(h12 * pk2 + N) - 2 * Q2 * np.log2(h21 * pk1 + N)  ###
            deltaG = [-2 * Q2 * h21 / ((h21 * pk1 + N) * cp.log(2)),
                      -2 * Q1 * h12/ ((h12 * pk2 + N) * cp.log(2))]  ###
            vectorP = [p[0] - pk1, p[1] - pk2] ###
            # object1 = cp.Minimize(f - g - deltaG*vectorP)
            object3 = cp.Minimize(f - g - deltaG[0]*vectorP[0]-deltaG[1]*vectorP[1])  ###
            constr3 = [p_min <= p, p <= p_max, A *p-B<=0]
            # constr1= [p_min <= p]
            # constr2=[p <= p_max]
            # constr3=[A * p + B <= p_min]
            # prob = cp.Problem(object, constr1+constr2+constr3)
            prob3 = cp.Problem(object3, constr3)
            prob3.solve(solver='SCS')
            if prob3.status == 'optimal' or prob3.status == 'optimal_inaccurate':
                pp1 = max(p.value[0],0)
                pp2 = max(p.value[1],0)
                # print('pp1,pp2', pp1, pp2)
                if phi1!=0 and pp1==0:
                    pp1 = 0
                    pp2 = 0
                    objcr= 1000
                    Rk1 = 0
                    Rk2 = 0

                    break
                if phi2!=0 and pp2==0:
                    pp1 = 0
                    pp2 = 0
                    objcr = 1000
                    Rk1 = 0
                    Rk2 = 0
                    break
                crk1=h11 * pp1 / (N + h12 * pp2)
                crk2= h22 * pp2 / (N + h21 * pp1)
                # print('crpp1,crpp2',pp1,pp2)
                # if crk1>=0 and crk2>=0:
                Rk1 = math.log2(1 + crk1)
                Rk2 = math.log2(1 +crk2)
                # else:
                #     Rk2=0
                #     Rk1=0
                # print('crRk1,crRk2', Rk1, Rk2)
                # print('p1,pk1',p.value[0],pk1)
                # print('p2,pk2', p.value[1], pk2)

                if abs(objcr-object3.value)<=epsilon1:
                    objcr=object3.value
                    break
                elif I>=20:
                    objcr = object3.value
                    break
                else:
                    objcr = object3.value
                    pk1 = pp1
                    pk2 = pp2
            else:
                pp1 = 0
                pp2 = 0
                objcr= 1000
                Rk1 = 0
                Rk2 = 0
                break
    else:
        pp1 = 0
        pp2 = 0
        objcr = 1000
        Rk1 = 0
        Rk2 = 0

    return objcr, pp1, pp2, Rk1, Rk2

r"""
tempobj, tp1, tp2, tempR1, tempR2 = \
    NOA(h11, h12, h21, h22, V, Q1, Q2, tempr1, tempr2, N0, phi1, phi2, pmax, eta, Y1, Y2)
    phi1, phi2是1，0
"""
def NOA(h11, h12, h21, h22, v, QQ1, QQ2, r1, r2, N, phii1, phii2, pma, eta, YY1, YY2):
    epsilon1 = 0.01     # 退出迭代门限
    k = (2 ** r1 - 1) * (2 ** r2 - 1) * h12 * h21 / (h11 * h22)     
    pnot1 = N * ((2 ** r1 - 1) * h21 / h11 + k) / (h21 * (1 - k))   # 需要的最小功率, FIXME: N是功率，不是功率密度
    pnot2 = N * ((2 ** r2 - 1) * h12 / h22 + k) / (h12 * (1 - k))
    # print(B)
    p_max = np.array([pma, pma])
    p_min = np.array([0, 0])
    # p_max=cp.Parameter
    # print('pnot1,pnot2',pnot1,pnot2)
    # print('p_max', p_max)

    if k < 1 and pnot1 <= pma and pnot2 <= pma:
        I = 0       # 迭代次数
        objNOA = -10000

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
        # pk1 = pnot1
        # pk2 = pnot2

        while True:
            I += 1
            A = np.array([[-1, h12 * (2 ** r1 - 1) / h11], [h21 * (2 ** r2 - 1) / h22, -1]])    # 2 × 2
            B = np.array([[-(2 ** r1 - 1) * N / h11], [-(2 ** r2 - 1) * N / h22]])              # 2 × 1
            p = cp.Variable(shape=(2, 1), nonneg=True)                                          # 1 × 2, np.dot(A, p) - B
            f = (v*alpha1+2*QQ1) * np.log2(math.e) * cp.log(N+h11*p[0]+h12*p[1]) + \
                (v*alpha2+2*QQ2) * np.log2(math.e) * cp.log(N+h22*p[1]+h21*p[0]) - \
                (v*eta*beta1) * p[0] - (v*eta*beta2) * p[1] - 2 * ((QQ1*r1-phii1*YY1) + (QQ2*r2-phii2*YY2)) # u_i, v_i = 1, 1
            y1 = N + h12 * pk2
            y2 = N + h21 * pk1
            # print('pk1,pk2',pk1,pk2)
            # print('N0',N)
            # if y1>(10**(-300)) and y2>(10**(-300)):########
            g = (v*alpha1+2*QQ1) * np.log2(y1) + (v*alpha2+2*QQ2) * np.log2(y2)
            # else:
            #     pp1 = 0
            #     pp2 = 0
            #     objNOA.value = -10000
            #     Rk1 = 0
            #     Rk2 = 0
            #     break
            vectorP = [p[0] - pk1, p[1] - pk2]
            deltaG = [(v*alpha2 + 2 * QQ2) * h21 / ((N + h21 * pk1) * cp.log(2)),
                      (v*alpha1 + 2 * QQ1) * h12 / ((N + h12 * pk2) * cp.log(2))]
            # print('G',type(deltaG),deltaG)
            # print('P',type(vectorP),vectorP)
            # objfunc1 = cp.Maximize(f-g-deltaG[0]* vectorP[0]-deltaG[1]* vectorP[1])
            objfunc4 = cp.Maximize(f - g - np.dot(deltaG, vectorP))  
            constr4 = [p_min <= p, p <= p_max, (np.dot(A, p) - B) <= 0 ]      # 原文公式(21)
            prob4 = cp.Problem(objfunc4, constr4)
            prob4.solve(solver='SCS')
            if prob4.status =='optimal' or prob4.status =='optimal_inaccurate':
                pp1 = max(p.value[0], 0)
                pp2 = max(p.value[1], 0)
                k1 = h11 * pp1 / (N + h12 * pp2)
                k2 = h22 * pp2 / (N + h21 * pp1)
                # print('pp1,pp2',pp1,pp2)
                if k1>=0 and k2>=0:
                    Rk1 = np.log2(1 + k1)
                    Rk2 = np.log2(1 + k2)
                else:
                    pp1 = 0
                    pp2 = 0
                    objNOA = prob4.value
                    Rk1 = 0
                    Rk2 = 0
                # print('pp1,pp2',pp1,pp2)
                # print('rk1,rk2',Rk1,Rk2)
                if abs(objNOA-objfunc4.value) <= epsilon1:
                    objNOA = objfunc4.value
                    break
                elif I >= 20:
                    objNOA = objfunc4.value
                    break
                else:
                    pk1 = pp1
                    pk2 = pp2
                    objNOA = objfunc4.value
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

