import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from Calculate import Pmin, TempP, TempP_EE, obj, obj_EE, NOA, crosslayer

np.random.seed(0)

global R1, R2, R_1, R_2, Rvv_1, fr_r1, fr_r2

phi = [[1,1], [1,0], [0,1], [0,0]] # 4个动作

phi_th = 0.9    # ty1门限
# VV=[2, 10]
alpha1 = 0.5      # 两个用户的权重
alpha2 = 0.5
beta1 = 0.5
beta2 = 0.5

dii = 10          # 距离
d21 = 100
d12 = 120
N0 = 10**(-15)  # 白噪声
band_width = 1e5
PLii = - (103.4 + 24.2 * np.log10(dii / 1000))
PL12 = - (103.4 + 24.2 * np.log10(d21 / 1000))
PL21 = - (103.4 + 24.2 * np.log10(d12 / 1000))
sigmaii = (10**(PLii / 10)) / (band_width * N0)      # TODO: 和ppo的环境改
sigma12 = (10**(PL12 / 10)) / (band_width * N0)      
sigma21 = (10**(PL21 / 10)) / (band_width * N0)      

T = 3000    # 3000步

VV=np.arange(2,4,3)     # 能效比在优化目标中的权重

lamda = 0.8               # ty2 generate rate
#原1.2 3.2 V1，1，13
r = [0, 0.2]               # ty1 transmission rate
pmax_dbm = np.arange(30, 31, 3)     # power, dBm
pavg_dbm = 20
pavg = (10 ** (pavg_dbm / 10)) / 1000
pnum = len(pmax_dbm)  # pnum = 1

EE = [[],[],[]] #energy efficiency
AvgsumPower = [[],[],[]]
Q = [[],[],[]]      # ty2 queue
Y = [[],[],[]]      # ty1 queue
REL = [[], [], []]      # 实际发没发送，每个slot都发是2，都不发是0，记录平均值
AvgsumRate = [[], [], []]

power_action = []
rate_action = []
for vnum in range(len(VV)):
    V = VV[vnum]
    # print('V',V)
    for a in range(pnum):
        P_max = pmax_dbm[a]
        pmax = (10 ** (P_max / 10)) / 1000      # dBm -> W
        # print('pmax_dbm',P_max)
        Q1 = 0.0 
        Q2 = 0.0
        Y1 = 0.0
        Y2 = 0.0
        eta = 0.000     # ee
        Rv1 = 0.0       # save rate
        Pv1 = 0.0       # save power
        sumQ = 0.0
        sumY = 0.0
        rel = 0.0       # 存实际选的动作，每一轮求和
        for t in range(T):
            # print('t', t)
            h11 = np.random.gamma(2, sigmaii / 2)  
            h22 = np.random.gamma(2, sigmaii / 2)
            h12 = np.random.gamma(2, sigma12 / 2)
            h21 = np.random.gamma(2, sigma21 / 2)

            a1 = np.random.poisson(lam=lamda)  # type-2产生速率
            a2 = np.random.poisson(lam=lamda)

            OBJ = -1000     # 先设一个小的值，A、B、C、D四种情况哪种大就选哪种
            R1 = 0
            R2 = 0
            P1 = 0
            P2 = 0
            b1_ee = 0         # ty2这一步的发送速率
            b2_ee = 0
            PHI1 = 0
            PHI2 = 0

            for i in range(4):  
                phi1 = phi[i][0]    # 选到这个任务要发送，那么就=1，否则=0
                phi2 = phi[i][1]
                tempr1 = r[phi1]    # 根据Control action的不同，对type-1发送优先级不同，psi=0 时 r=0 else r=0.2
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
            # print('p1,p2',P1,P2)
            # print('R1,R2',R1,R2)
            # 一个时刻t选定一种方案
            sumQ += (Q1 + Q2)
            sumY += (Y1 + Y2)
            Q1 = max(Q1 - b1_ee, 0) + a1
            Q2 = max(Q2 - b2_ee, 0) + a2
            Y1 = max(Y1 - PHI1, 0) + phi_th
            Y2 = max(Y2 - PHI2, 0) + phi_th
            # print('a1,a2',a1,a2)
            # print('Q1,Q2',Q1,Q2)

            Rv1 += (alpha1 * R1 + alpha2 * R2)
            Pv1 += (beta1 * P1 + beta2 * P2)

            if Pv1 == 0.0:
                eta = 0.0
            else:
                eta = Rv1 / Pv1
            rel += (PHI1 + PHI2)    

        if Pv1 == 0.0:
            EE[vnum].append(0.0)    # 在EE的第一维附加上去
        else:
            EE[vnum].append(round((Rv1 / Pv1), 3))   # 四舍六入

        AvgsumPower[vnum].append(((Pv1) / T))
        Q[vnum].append(round((sumQ) / T, 3))
        REL[vnum].append(round(rel / (2 * T), 3))
        AvgsumRate[vnum].append(round(Rv1 / T, 3))
        Y[vnum].append(round((sumY) / T, 3))
        # print('EE', EE)  
        # print('AvgsumPower', AvgsumPower)  
        # print('Q', Q)
        # print('AvgRate', AvgsumRate)
        # print('rel', REL)

# print('EE',EE) 
# print('AvgsumPower',AvgsumPower) 
# print('Q',Q)
# print('AvgRate',AvgsumRate)
# print('rel',REL)
# print('Y',Y)

fig1 = plt.figure("0420 23 36Achievable energy efficiency")  #颜色选一下 两个图 #横坐标和lengend可不可以有下标
PE1 = plt.semilogy(pmax_dbm, EE[0],color='blue', marker='o',label='EE-based design,V=2')
PE2 = plt.semilogy(pmax_dbm, EE[1],color='orange', marker='*',label='EE-based design,V=5')
PE3 = plt.semilogy(pmax_dbm, EE[2],color='red', marker='o',label='EE-based design,V=8')

plt.xlabel("pmax_dbm")
plt.ylabel("Energy Efficiency")
plt.legend(loc='upper right')
# #
fig2 = plt.figure("Reliability of type-1 messages ")

PR1 = plt.plot(pmax_dbm, REL[0], color='blue', linestyle='-',label='EE-based design,V=1')
PR2 = plt.plot(pmax_dbm, REL[1], color='orange', linestyle='-.',label='EE-based design,V=4')
PR3 = plt.plot(pmax_dbm, REL[2], color='red', linestyle='-',label='EE-based design,V=7')
# PR4 = plt.plot(V, REL[3], color='black', linestyle='-.',label='EE-based design,pmax=25')
# PR5 = plt.plot(V, REL[4], color='green', linestyle='-',label='EE-based design,pmax=30')
plt.xlabel("pmax_dbm")
plt.ylabel("Reliability")
plt.legend(loc='upper right')
fig3= plt.figure("queue length of type-2")
PQ1 = plt.plot(pmax_dbm, Q[0], color='blue', linestyle='-.',label='EE-based design,V=1')
PQ2 = plt.plot(pmax_dbm, Q[1], color='orange', linestyle='-.',label='EE-based design,pmax=15')
PQ3 = plt.plot(pmax_dbm, Q[2], color='red', linestyle='-.',label='EE-based design,pmax=23')
# PQ4 = plt.plot(V, Q[3], color='black', linestyle='-.',label='EE-based design,pmax=25')
# PQ5 = plt.plot(V, Q[4], color='green', linestyle='-.',label='EE-based design,pmax=30')
plt.xlabel("pmax_dbm")
plt.ylabel("Queue length")
plt.legend(loc='upper right')
# PQ3 = ax2.plot(pmax_dbm, fr_Q,color='red', linestyle='-.',label='CSI-based design')
# PQ4 = ax2.plot(pmax_dbm, Q_,color='green', linestyle='-.',label='PM-based design')
# PQ5 = ax2.plot(pmax_dbm, Q,color='black', linestyle='-.',label='OT-based design')

fig4 = plt.figure("AvgsumRate")  #颜色选一下 两个图 #横坐标和lengend可不可以有下标
PRA1 = plt.plot(pmax_dbm, AvgsumRate[0],color='blue', linestyle='-',label='EE-based design,V=1')
PRA2 = plt.plot(pmax_dbm, AvgsumRate[1],color='orange', linestyle='-',label='EE-based design,V=4')
PRA3 = plt.plot(pmax_dbm, AvgsumRate[2],color='red', linestyle='-',label='EE-based design,V=7')
# PRA4 = plt.plot(V, AvgsumRate[3],color='black', linestyle='-',label='EE-based design,pmax=25')
# PRA5 = plt.plot(V, AvgsumRate[4],color='green', linestyle='-',label='EE-based design,pmax=30')
plt.xlabel("pmax_dbm")
plt.ylabel("AvgsumRate")
plt.legend(loc='upper right')

fig5 = plt.figure("AvgsumPower")  #颜色选一下 两个图 #横坐标和lengend可不可以有下标
VP1=plt.plot(pmax_dbm,AvgsumPower[0],color='blue', linestyle='--',label='EE-based design,V=1')
VP2=plt.plot(pmax_dbm,AvgsumPower[1],color='orange', linestyle='--',label='EE-based design,V=4')
VP3=plt.plot(pmax_dbm,AvgsumPower[2],color='red', linestyle='-',label='EE-based design,V=7')
# VP4=plt.plot(V,AvgsumPower[3],color='black', linestyle='-',label='EE-based design,pmax=25')
# VP5=plt.plot(V,AvgsumPower[4],color='green', linestyle='-',label='EE-based design,pmax=30')
plt.xlabel('pmax_dbm')
plt.ylabel('AvgsumPower')
# VP1 = plt.semilogy(V, AvgsumPower,color='black', marker='o',label='EE-based design,V=2')
# plt.xlabel("V")
# plt.ylabel("AvgsumPower")
plt.legend(loc='upper right')
plt.show()


