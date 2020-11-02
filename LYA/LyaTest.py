r"""
Original Paper: D. Lan, C. Wang, P. Wang, F. Liu and G. Min, "Transmission Design for Energy-Efficient 
Vehicular Networks with Multiple Delay-Limited Applications," 2019 IEEE Global Communications Conference (GLOBECOM), 
Waikoloa, HI, USA, 2019, pp. 1-6.
doi: 10.1109/GLOBECOM38437.2019.9014246
keywords: {Optimization;Sensors;Reliability;Delays;Power demand;Encoding;Linear programming},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9014246&isnumber=9013108

Windows 10: cvxpy 1.0.21 and Ubuntu 18.04: cvxpy 1.0.27 have been tested.
"""


import copy
import datetime
import os

import cvxpy as cp
import numpy as np
from tqdm import tqdm

# np.random.seed(0)

def NOA(h11, h12, h21, h22, v, QQ1, QQ2, r1, r2, N, phii1, phii2, pma, eta, YY1, YY2):
    """
    non-orthogonal
    """
    cpln2 = 0.6931471805599453              # ln2 = 0.6931471805599453
    epsilon = 0.0001                        # iteration breaking threshold
    itera_max = 100

    amp = 1
    N *= amp

    k = (2 ** r1 - 1) * (2 ** r2 - 1) * h12 * h21 / (h11 * h22)     
    pnot1 = N * ((2 ** r1 - 1) * h21 / h11 + k) / (h21 * (1 - k))       # minimum power required
    pnot2 = N * ((2 ** r2 - 1) * h12 / h22 + k) / (h12 * (1 - k))
    p_max = np.array([[pma], [pma]])                # 2 × 1
    p_min = np.array([[pmin], [pmin]])              # 2 × 1
    # p_min = np.array([[0], [0]])    

    if k < 1 and pnot1 <= pma and pnot2 <= pma:
        I = 0                               # iteration num
        objNOA = -10000
        pk1 = pma
        pk2 = pma
        while True:
            I += 1
            p = cp.Variable(shape=(2, 1), nonneg=True)      # 2 × 1, np.dot(A, p) - B
            # if use "f = (...) / cp.log(2)", will raise error "Problem does not follow DCP rules."
            f = (v * alpha1 + 2 * u1 * QQ1) * cp.log(N + h11 * p[0][0] * amp + h12 * p[1][0] * amp) / cpln2 \
                + (v * alpha2 + 2 * u2 * QQ2) * cp.log(N + h22 * p[1][0] * amp + h21 * p[0][0] * amp) / cpln2 \
                - (v * eta * beta1) * p[0][0] - (v * eta * beta2) * p[1][0] \
                - 2 * (u1 * QQ1 * r1 - phii1 * v1 * YY1 + u2 * QQ2 * r2 - phii2 * v2 * YY2)                           
            y1 = N + h12 * pk2 * amp
            y2 = N + h21 * pk1 * amp
            g = (v * alpha1 + 2 * u1 * QQ1) * cp.log(y1) / cpln2 + (v * alpha2 + 2 * u2 * QQ2) * cp.log(y2) / cpln2

            vectorP = np.array([p[0][0] - pk1, p[1][0] - pk2])
            deltaG = np.array([(v * alpha2 + 2 * u2 * QQ2) * h21 / ((N + h21 * pk1) * cpln2),
                               (v * alpha1 + 2 * u1 * QQ1) * h12 / ((N + h12 * pk2) * cpln2)])

            A = np.array([[-1, h12 * (2 ** r1 - 1) / h11], [h21 * (2 ** r2 - 1) / h22, -1]])    # 2 × 2
            B = np.array([[-(2 ** r1 - 1) * N / h11], [-(2 ** r2 - 1) * N / h22]])              # 2 × 1                   

            objfunc = cp.Maximize(f - g - deltaG[0] * vectorP[0] - deltaG[1] * vectorP[1])  
            constr = [p_min <= p, p <= p_max, (A * p - B) <= 0 ]
            prob = cp.Problem(objfunc, constr)
            # print(prob)
            prob.solve(solver = cp.SCS)
            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                pp1 = max(p.value[0][0], pmin)
                pp2 = max(p.value[1][0], pmin)
                pp1 = min(p.value[0][0], pma)
                pp2 = min(p.value[1][0], pma)
                k1 = h11 * pp1 / (N + h12 * pp2)        # SINR
                k2 = h22 * pp2 / (N + h21 * pp1)
                if k1 >= 0 and k2 >= 0:
                    Rk1 = np.log2(1 + k1)               # channel capacity, bps
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
                elif I >= itera_max:
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


##############################################################################
actions = [[1, 1]]

phi_th = 0.9                            # ty1 transmission success shreshold

alpha1 = 0.5                            # rate weight
alpha2 = 0.5
beta1 = 0.5                             # power weight      
beta2 = 0.5

N0 = 1
band_width = 1                      

T = 300                                 # time slot

r = [0, 0.5]                            # ty1 transmission rate
lamda = 1                               # ty2 generate rate

ALL_V = [3]                             # weight of energy efficiency
v1, v2 = 1, 1                           # ty1 queue weight
u1, u2 = 0.3, 0.3                       # ty2 queue weight

pmin = 0
pmax = 100

scale_ii = np.sqrt(1 / 2)
scale_ij = np.sqrt(0.03 / 2)

real_rmax = [0, 0]

EE = []                                 # energy efficiency
AvgPower = []
AvgRate = []
avg_I = []
Y = []                                  # ty1 queue
Q = []                                  # ty2 queue
RELI = []                               # reliability of ty1
large_than_Q = 0
all_large_Q = []
lower_than_phi = 0

path = './Lyapunov_results/'
if not os.path.exists(path):
    os.makedirs(path)
file_name = path + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'

eps = 33
with open(file_name, 'w') as fi:
    for ep in range(eps):
        bad_ep = False
        exp = []
        for vnum in range(len(ALL_V)):
            V = ALL_V[vnum]                     
            Y1 = 0.0
            Y2 = 0.0
            Q1 = 0.0 
            Q2 = 0.0
            real_Qmax_1 = 0
            real_Qmax_2 = 0
            eta = 3                             # ee
            eta_init = copy.deepcopy(eta)
            sumR = 0.0                          # sum rate
            sumI = 0.0
            sumP = 0.0                          # sum power
            sumQ = 0.0
            sumY = 0.0
            reli = 0.0
            for t in tqdm(range(T), desc=f'Epoch #{ep+1}', ncols=60):
                h11 = (np.random.rayleigh(scale_ii)) ** 2
                h22 = (np.random.rayleigh(scale_ii)) ** 2
                h12 = (np.random.rayleigh(scale_ij)) ** 2
                h21 = (np.random.rayleigh(scale_ij)) ** 2
                # print(h11, h12, h21)

                a1 = np.random.poisson(lam = lamda)     # ty2 data
                a2 = np.random.poisson(lam = lamda)

                OBJ = -1e6                              # a small initial objective 
                R1 = 0
                R2 = 0
                I1 = 0
                I2 = 0
                P1 = 0
                P2 = 0
                b1 = 0                                  # ty2 transmission rate
                b2 = 0
                PHI1 = 0                                # the best of the four actions
                PHI2 = 0

                # fi.write('\nslot:{:d}, --Y1:{:.4f}, --Q1:{:.4f},'.format(t+1, Y1, Q1))
                # exp.append(' ')
                # exp.append(['slot', t + 1, 'Y1', Y1, 
                #     'Q1', Q1*band_width, 'Y2', Y2, 'Q2', Q2*band_width])
                # exp.append(['hij', h11, h22, h12, h21])
                # exp.append(['ty2 generate/kbps', a1*band_width, a2*band_width])
                real_Qmax_1 = Q1 if Q1 > real_Qmax_1 else real_Qmax_1
                real_Qmax_2 = Q2 if Q2 > real_Qmax_2 else real_Qmax_2
                if Q1 > 10*lamda / band_width or Q2 > 10*lamda / band_width:
                    large_than_Q += 1
                    bad_ep = True
                    # print('\nlarge than Qmax!', Q1 * band_width, Q2 * band_width)
                    # for i in exp:
                    #     print(i)
                for i in range(len(actions)):
                    phi1 = actions[i][0]                # transfer ty1 or not
                    phi2 = actions[i][1]
                    tempr1 = r[phi1]
                    tempr2 = r[phi2]

                    tempobj, temp1, temp2, tempR1, tempR2 \
                        = NOA(h11, h12, h21, h22, V, Q1, Q2, 
                        tempr1, tempr2, N0, phi1, phi2, pmax, eta, Y1, Y2)

                    if tempobj > OBJ:
                        OBJ = tempobj
                        P1 = temp1 + 0.0001
                        P2 = temp2 + 0.0001
                        real_rmax[0] = tempR1 if tempR1 > real_rmax[0] else real_rmax[0]
                        real_rmax[1] = tempR1 if tempR1 > real_rmax[1] else real_rmax[1]
                        R1 = tempR1 if tempR1 < Q1 + r[1] else Q1 + r[1]
                        R2 = tempR2 if tempR2 < Q2 + r[1] else Q2 + r[1]
                        I1 = tempR1
                        I2 = tempR2
                        # print('\n', tempobj, P1, P2, tempR1, tempR2)
                        # print(np.log2(1 + P1 * h11 / (P2 * h12 + N0)), 
                        #     np.log2(1 + P2 * h22 / (P1 * h21 + N0)))
                        if R1 >= tempr1:
                            PHI1 = phi1
                            b1 = R1 - tempr1
                        else:
                            b1 = 0
                        if R2 >= tempr2:
                            PHI2 = phi2
                            b2 = R2 - tempr2
                        else:
                            b2 = 0
                # if OBJ == -1e6:
                #     print(tempobj)
                #     print(h11, h12, h21, h22)
                #     print(PHI1, PHI2)
                #     for i in exp:
                #         print(i)
                #     assert False, 'Unsolvable!'
                # action_counter[str([PHI1, PHI2])] += 1
                sumY += (Y1 + Y2)
                sumQ += (Q1 + Q2)
                Y1 = max(Y1 - PHI1, 0) + phi_th     # ty1 queue
                Y2 = max(Y2 - PHI2, 0) + phi_th
                Q1 = max(Q1 - b1, 0) + a1           # ty2 queue
                Q2 = max(Q2 - b2, 0) + a2
                sumP += (beta1 * P1 + beta2 * P2)
                sumR += (alpha1 * R1 + alpha2 * R2)
                sumI += (alpha1 * I1 + alpha2 * I2)
                if sumP == 0.0:
                    eta = 0.0
                else:
                    eta = sumR / sumP
                reli += (PHI1 + PHI2)

                # fi.write('''\n--power:{:.8f}, --rate:{:.8f}, --ee:{:.4f}, 
                #     --trans_ty1_rate:{:.1f}, --trans_ty2_rate:{:.8f}\n'''
                #     .format(P1, R1, eta, PHI1, b1))
                # exp.append([P1, P2])
                # exp.append(['R1/kbps', R1*band_width, 'R2/kbps', R2*band_width])
                # exp.append(["ee", eta])
                # exp.append(['transmit', 'ty1 rate of 1/kbps', r[1]*band_width, 
                #     'ty2 rate of 1/kbps', b1*band_width])
                # exp.append(['transmit', 'ty1 rate of 2/kbps', r[1]*band_width, 
                #     'ty2 rate of 2/kbps', b2*band_width])
            # print('action counter:', action_counter)
            if sumP == 0.0:
                EE.append(0.0)
            else:
                EE.append(sumR / sumP)
            AvgPower.append(sumP / T)
            AvgRate.append(sumR / T)
            avg_I.append(sumI / T)
            Y.append(sumY / (2 * T))
            Q.append(sumQ / (2 * T * lamda / band_width))
            RELI.append(reli / (2 * T))
            if RELI[-1] < phi_th:
                lower_than_phi += 1
        print('Num of time slot which Q is larger than Q_max:', large_than_Q, 
            '\nNum of episode which phi is lower than phi_th:', lower_than_phi)
        print('Q_max_1, Q_max_2:', real_Qmax_1, real_Qmax_2)
        all_large_Q.append([real_Qmax_1, real_Qmax_2])
        print('avg EE:', sum(EE)/len(EE))
        print('avg reli:', sum(RELI)/len(RELI), '\n')
        # if True:
        # if bad_ep:
        #     for i in exp:
        #         print(i)
        #     print('')

    fi.write('ALL_V: ' + str(ALL_V) + '\n')
    fi.write('EE initial: ' + str(eta_init) + '\n')
    fi.write('ty1, 2 data: ' + str(r[1]*band_width) + ', ' + str(lamda) + '\n')
    fi.write('v1/v2: ' + str(v1) + ', ' + 'u1/u2: ' + str(u1) + '\n')
    fi.write('Energy Efficiency: ' + str(EE) + '\n')
    fi.write('Y: ' + str(Y) + '\n')
    fi.write('Q: ' + str(Q) + '\n')
    fi.write('RELI: ' + str(RELI) + '\n')
    fi.write('AvgPower: ' + str(sum(AvgPower)/len(AvgPower)) + '\n')
    fi.write('AvgRate: ' + str(AvgRate) + '\n')

print('EE: ', sum(EE)/len(EE), 
    '\nY: ', sum(Y)/len(Y), 
    '\nQ: ', sum(Q)/len(Q), ' / ', sum(Q)/len(Q) * lamda, 
    '\nRELI: ', sum(RELI)/len(RELI), 
    '\navg power: ', sum(AvgPower)/len(AvgPower), 
    '\navg rate: ', sum(AvgRate)/len(AvgRate) * band_width, 
    '\navg channel capacity: ', sum(avg_I)/len(avg_I) * band_width, '\n', sep='')
print('Num of time slot which Q is larger than Q_max:', large_than_Q)
print('Max Q of each episode:', all_large_Q)
print('rmax/Hz:', real_rmax)
