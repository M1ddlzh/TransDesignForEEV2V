import math
import cvxpy as cp
import numpy as np


share_para = 1 / 3
alpha1=share_para
alpha2=share_para
alpha3=share_para
beta1=share_para
beta2=share_para
beta3=share_para
beta0=0
epsilon1=0.0001
cpln2 = 0.6931471805599453              # ln2 = 0.6931471805599453


def Pmin(rr1, rr2, rr3, delt, h1, h2, h3, N):
    pmi1 = (2**(rr1/delt)-1)*N/h1
    pmi2 = (2**(rr2/delt)-1)*N/h2
    pmi3 = (2**(rr3/delt)-1)*N/h3
    return pmi1, pmi2, pmi3


def TempP_EE(v, Q1, Q2, Q3, Z1, Z2, Z3, delt, eta, N, h11, h22, h33, pmi1, pmi2, pmi3, pma):
    pp1 = delt * ((2*Q1)/(math.log(2)*(2*Z1+v*beta1))-N/h11)
    pp2 = delt * ((2*Q2)/(math.log(2)*(2*Z2+v*beta2))-N/h22)
    pp3 = delt * ((2*Q3)/(math.log(2)*(2*Z3+v*beta3))-N/h33)
    p11 = min(max(pp1, pmi1), pma)
    p22 = min(max(pp2, pmi2), pma)
    p33 = min(max(pp3, pmi3), pma)
    return p11, p22, p33


def obj_EE(v, delt, h11, h22, h33, p1, p2, p3, N, eta, Z1, Z2, Z3, 
    Q1, Q2, Q3, Y1, Y2, Y3, rr1, rr2, rr3, phi1, phi2, phi3):
    RR1 = delt*math.log2(1+h11*p1/N)
    RR2 = delt*math.log2(1+h22*p2/N)
    RR3 = delt*math.log2(1+h33*p3/N)
    Tempobj = (v*(alpha1*RR1+alpha2*RR2+alpha3*RR3
            -eta*beta1*p1-beta2*eta*p2-beta3*eta*p3)
        -2*(Z1*p1+Z2*p2+Z3*p3
            +Q1*rr1+Q2*rr2+Q3*rr3
            -Y1*phi1-Y2*phi2-Y3*phi3)
        +2*(Q1*RR1+Q2*RR2+Q3*RR3))
    return Tempobj,RR1,RR2,RR3
