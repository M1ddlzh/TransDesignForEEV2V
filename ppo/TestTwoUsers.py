import numpy as np
import matplotlib.pyplot as plt

dii = 10        # m
d21 = 100
d12 = 120
n0 = 10**-15    # W/Hz
band_width = 1e5    # Hz

plii = -(103.4+24.2*np.log10(dii/1000))     # Km, dB
pl12 = -(103.4+24.2*np.log10(d21/1000))
pl21 = -(103.4+24.2*np.log10(d12/1000))
print(plii, pl12, pl21)

sigmaii = (10**(plii/10))   # 经过大尺度衰落后功率缩小多少倍，期望
sigma12 = (10**(pl12/10))
sigma21 = (10**(pl21/10))
print(sigmaii, sigma12, sigma21)

all_h11, all_h22, all_h12, all_h21 = [], [], [], []
m = 8
for _ in range(20000):
    h11 = np.random.gamma(m, sigmaii / m)   # 期望除以参数m作为gamma分布的伸缩因子，因为两个参数相乘应等于期望
    h22 = np.random.gamma(m, sigmaii / m)
    h12 = np.random.gamma(m, sigma12 / m)
    h21 = np.random.gamma(m, sigma21 / m)
    all_h11.append(h11)
    all_h22.append(h22)
    all_h12.append(h12)
    all_h21.append(h21)
all_h0 = np.array(all_h11)
all_h1 = np.array(all_h22)
all_h2 = np.array(all_h12)
all_h3 = np.array(all_h21)
print(np.max(all_h0), np.max(all_h1), np.max(all_h2), np.max(all_h3))
print(np.mean(all_h0), np.mean(all_h1), np.mean(all_h2), np.mean(all_h3))

_, axs = plt.subplots(2, 2, constrained_layout=True)
fig_i = 0
for ax in axs.flatten():
    ax.hist(eval('all_h' + str(fig_i)), bins=20)
    fig_i += 1
# plt.show()

p_min = 2       # dBm, 23~600k-1100k, 3~300k-700k
p_max = 3

snr = []
i1, i2 = [], []
for _ in range(100000):
    h11 = np.random.gamma(2, sigmaii / 2)       # 信道参数h服从Nakagami-m分布时，h**2服从伽马分布，这里的h**2就是hii
    h22 = np.random.gamma(2, sigmaii / 2)
    h12 = np.random.gamma(2, sigma12 / 2)
    h21 = np.random.gamma(2, sigma21 / 2)
    p1 = 10 ** (np.random.uniform(p_min, p_max) / 10) / 1000        # 变成 W
    p2 = 10 ** (np.random.uniform(p_min, p_max) / 10) / 1000
    snr.append(h11 * p1 / (band_width * n0 + h12 * p2))
    i1.append(band_width * np.log2(1 + h11 * p1 / (band_width * n0 + h12 * p2)))
    i2.append(band_width * np.log2(1 + h22 * p2 / (band_width * n0 + h21 * p1)))

_, axs = plt.subplots(1, 2, constrained_layout=True)
fig_i = 1
for ax in axs.flatten():
    ax.hist(eval('i' + str(fig_i)), bins=20)
    fig_i += 1
plt.show()
print(10 * np.log10(sum(snr) / 100000))     # dB
# 均值在600kbps，1000kbps以上就比较少了


