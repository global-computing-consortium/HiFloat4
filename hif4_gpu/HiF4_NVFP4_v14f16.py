import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def To_MXF4(x, G = 32): 
    # E8 + E2M1, column-wise convert
    # Round Half tie to even
    # if overflow, clip to upper-bound of E2M1

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-1000)))          
            Emax = np.max(E) 
            E8 = Emax - 2       # Sub Max_Exp of E2M1

            if E8 < -127:      
                E8 = -127

            igv= tmp*2**(-E8)    # original in_grp values

            E2 = np.floor(np.log2(igv + 2**(-1000)))    
            E2[E2 < 0] = 0      # subnormal     
            M1 = np.round(igv*2**(-E2 + 1))*2**(-1)     # Round Half to Even
            E2M1 = 2**E2*M1
            E2M1[E2M1 > 6] = 6  # Clip to Upper-bound

            grp = S * E2M1 * 2**E8

            res[i*G:i*G+G,j] = grp

    return res

def To_MXN(x, N = 4, G = 16): 
    # {E8 + 8*E1} + S1PN: Two-level scaling MX Format
    # EXP = [-127, 127], column-wise convert
    # MX4: N = 4, G = 16
    # MX6: N = 6, G = 16
    x = np.array(x)
    Mi = x.shape[0]
    Ni = x.shape[1]
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 3
    G2 = np.int32(G/8)
    for i in range(Mcnt):
        for j in range(Ni):
            ori = x[i*G:i*G+G,j]
            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-1000)))          
            Emax = np.max(E) 
            E8 = Emax - 1       # level-1 Scaling: [-127, 127],127 for NaN

            E1x8 = np.zeros(8)  # level-2 Scaling
            for k in range(8):
                E1x8[k] = np.max(E[k*G2:k*G2+G2])                    
            E1x8 = E1x8 - E8
            E1x8[E1x8 < 0] = 0  # [0, 1]

            E8G = E1x8 + E8     # Fused 8 scaling exp
            EG = np.zeros(G)    # Fused G scaling exp
            for k in range(8):
                EG[k*G2:k*G2+G2] = E8G[k]

            in_grp = np.floor(tmp*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            in_grp[in_grp >= 2] = 2- 2**(-Ng)
            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res

def To_NVF4(x, G = 16): 
    # E4M3 + E2M1, column-wise convert
    # Round Half tie to even
    # if overflow, clip to upper-bound of E2M1
    x = np.array(x)
    Mi = x.shape[0]
    Ni = x.shape[1]   
    Mcnt = np.ceil(Mi/G).astype(int)
    res = np.zeros((Mi, Ni))
    for i in range(Mcnt):
        for j in range(Ni):
            ori = x[i*G:i*G+G,j]
            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            # 1 DIV-1 ---> replace by (x * 1/6)
            SF = np.max(tmp) / 6  # Scale Factor = Max_Value / 6 (6 is the max number represented by E2M1)   
            SF = 448 if (SF > 448) else SF    # overflow handling
            E_SF = np.floor(np.log2(SF + 2**(-1000)))    # Exp of E4M3
            E_SF = -6 if (E_SF < -6) else E_SF    # subnormal handling
            E4M3 = np.round(SF * 2 ** (-E_SF + 3)) * 2 ** (-3 + E_SF)

            # 16 DIV-2
            igv= tmp / (E4M3 + 2**(-1000))    # original in_grp values
            E2 = np.floor(np.log2(igv + 2**(-1000)))    
            E2[E2 < 0] = 0      # subnormal     
            M1 = np.round(igv*2**(-E2 + 1))*2**(-1)     # Round Half to Even
            E2M1 = 2**E2*M1
            E2M1[E2M1 > 6] = 6  # Clip to Upper-bound
            grp = S * E2M1 * E4M3
            res[i*G:i*G+G,j] = grp
    return res

def To_BF16(x):
    x = np.float64(x)
    tmp = np.abs(x)
    E = np.floor(np.log2(tmp + 2**(-1000)))
    # E[E < -126] = -126
    E = -126 if E < -126 else E
    res = np.round(x*2**(-E + 7))*2**(E - 7)
    return res

def E6M2_REC(x):
    # input: E6M2, 8-bit
    # output: BF16, 16-bit
    # Compute: 1 / E6M2
    E6 = np.floor(np.log2(x + 2**(-1000)))     # x >= 0
    M2 = x * 2 ** (-E6 + 2) - 4

    if M2 == 0:     # 1.00
        M7 = 0.0        # binary: 000-0000
    elif M2 == 1:   # 1.25
        M7 = 77         # binary: 100-1101
    elif M2 == 2:   # 1.50
        M7 = 43         # binary: 010-1011
    elif M2 == 3:   # 1.75 
        M7 = 18         # binary: 001-0010
    else:
        print('Unexpected Input')
        exit()
    
    if M2 == 0:
        E8 = - E6
    else:
        E8 = - E6 - 1
    
    res = 2 ** (E8) * (1 + M7 * 2 ** (-7))
    return res

def BF16_to_E6MX(x, M = 2):
    # input Non-negative BF16
    # output E6M2
    # E6M2: 2^[-48, 15] * 1.M2, E6_offset = 48, 2^15 * 1.75 is NaN
    # E6M2_Max = 2^15 * 1.50 = 49152
    # E6M2_Min = 2^(-48)
    x = 2 ** 15 * 1.5 if (x > 2**15 * 1.5) else x       # overflow handling 
    x = 2 ** (-48) if (x < 2 ** (-48)) else x           # underflow handling
    E = np.floor(np.log2(x))                            # Exponent
    E6M2 = x * 2 ** (-E + M)
    E6M2 = np.round(E6M2)
    E6M2 = E6M2 * 2 ** (-M + E)  # Round Half tie to even or away
    return E6M2

def To_HiFX(x, N = 4, M = 2, G = 64): 
    # {E6M2 + 8*E1 + 16*E1} + S1PNg: Three-level scaling, column-wise convert
    # E6M2: 2^[-48, 15] * 1.M2, E6_offset = 48, 2^15 * 1.75 is NaN
    # E6M2_Max = 2^15 * 1.50 = 49152
    # E6M2_Min = 2^(-48)
    # Max absolute value of level-2&3 & igrp representation: 2^2 * 1.75 = 7
    # HiF4: N = 4, G = 64
    # HiF5: N = 5, G = 64
    x = np.array(x)
    Mi = x.shape[0]
    Ni = x.shape[1]
    Mcnt = np.ceil(Mi/G).astype(int)

    res    = np.zeros((Mi,     Ni))  # HiF4/5 
    GE6M2  = np.zeros((Mcnt,   Ni))  # level-1 Scale Factor
    GE1_8  = np.zeros((Mcnt*8, Ni))  # level-2 EXP 
    GE1_16 = np.zeros((Mcnt*16,Ni))  # level-3 EXP
    GDE16  = np.zeros((Mcnt*16,Ni))  # level-2 + level-3 EXP
    Sign   = np.zeros((Mi,     Ni))  # Sign of each datum
    igpA   = np.zeros((Mi,     Ni))  # in-grp absolute value
    VmaxS  = np.zeros((Mcnt,   Ni))  # Max in-grp value after 1.M2 scale
    
    Ng = N - 2
    for i in range(Mcnt):
        for j in range(Ni):
            ori = x[i*G:i*G+G,j]
            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values
            V16 = np.zeros(16)
            for k in range(16):
                V16[k] = np.max(tmpG[k*4:k*4+4])
            V8 = np.zeros(8)
            for k in range(8):
                V8[k] = np.max(V16[k*2:k*2+2])            
            Vmax = np.max(V8)

            # level-1 Scale Facor
            Const_rec = To_BF16(1 / 7.00)          # 7 is the max number represented by level-2&3 * S1P2
            SF = To_BF16(Vmax * Const_rec)         # BF16*BF16：Scale Factor = Max_Value * Const_rec
            E6M2 = BF16_to_E6MX(SF, M)

            # level-2 exp
            # REC_E6M2 = E6M2_REC(E6M2)       # special instructoin in hardware
            REC_E6M2 = To_BF16(1 / E6M2)    
            E1_8 = ((V8 * REC_E6M2) >= 4).astype(np.float32)     # [0, 1]

            # level-3 exp
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k*2:k*2+2] = E1_8[k]         
            E1_16 = ((V16 * REC_E6M2 * 2 ** (-E1_8x2)) >= 2).astype(np.float32)  # [0, 1]

            # v8x8 = np.zeros(64)
            # for iii in range(16):
            #     v8x8[iii*4:iii*4+4] = np.exp2(E1_16[iii]) * np.exp2(E1_8[iii//2])

            # Restore grp vals
            DE16 = E1_16 + E1_8x2             # Fused 16 Exp offsets
            DE64 = np.zeros(G)                # Fused 64 Exp offsets
            for k in range(16):
                DE64[k*4:k*4+4] = DE16[k]

            in_grp = tmpG * REC_E6M2 *2**(-DE64 + Ng)
            in_grp = np.floor(in_grp + 0.5)
            in_grp = in_grp * 2.0**(-Ng)
            # in_grp = np.floor(tmpG * REC_E6M2 *2**(-DE64 + Ng) + 0.5)*2.0**(-Ng)    # Round Half tie to away or even
            in_grp[in_grp >= 2] = 2- 2**(-Ng)       # Overflow Handling
            grp = S * E6M2 * 2.0 ** DE64 * in_grp

            res   [i*G:i*G+G,    j] = grp
            # GE6M2 [i,            j] = E6M2
            # GE1_8 [i*8:i*8+8,    j] = E1_8
            # GE1_16[i*16:i*16+16, j] = E1_16
            # GDE16 [i*16:i*16+16, j] = DE16
            # Sign  [i*G:i*G+G,    j] = S
            # igpA  [i*G:i*G+G,    j] = in_grp
            # VmaxS [i,            j] = Vmax * REC_E6M2 / 4

    return res#, GE6M2, GE1_8, GE1_16, GDE16, Sign, igpA, VmaxS

# ######--MSE CMP--######

# M = 1024
# N = 1024

# Tcnt = 18   # Test Number for Gaussian Dist
# print('Test for Gaussian Dist\n')

# HiF4_MSE = np.zeros(Tcnt)
# NVF4_MSE = np.zeros(Tcnt)
# NVF4_PTS_MSE = np.zeros(Tcnt)       # PTS: Per Tensor Scale
# MXF4_MSE = np.zeros(Tcnt)

# MSE_NORM_NVF4 = np.zeros(Tcnt)
# MSE_NORM_NVF4_PTS = np.zeros(Tcnt)
# MSE_NORM_MXF4 = np.zeros(Tcnt)

# for NT in range(Tcnt):
#     sigma = 1e-2 * 2 ** NT
#     mu = 0
#     x = np.random.normal(mu, sigma, (M, N))

#     HiF4, GE6M2, GE1_8, GE1_16, GDE16, Sign, igpA, VmaxS = To_HiFX(x)
#     MSE_HiF4 = np.sum((HiF4 - x)**2)/(M*N)
#     print(np.max(VmaxS))
#     print(np.min(VmaxS))
#     print(np.mean(VmaxS))
#     print(100*(1 - np.mean(VmaxS < 1.625)),'\n')

#     NVF4 = To_NVF4(x)
#     MSE_NVF4 = np.sum((NVF4 - x)**2)/(M*N)
#     MSE_NVF4_NORM = MSE_NVF4 / MSE_HiF4

#     PTS_SF = np.max(np.abs(x)) / 2688           # Per Tensor Scale Factor; 2688: max absolute value of NVF4
#     x_scale = x / PTS_SF                        # Scaled tensor 
#     NVF4_PTS = To_NVF4(x_scale) * PTS_SF        # Restored NVF4 tensor by PTS_SF
#     MSE_NVF4_PTS = np.sum((NVF4_PTS - x)**2)/(M*N)
#     MSE_NVF4_PTS_NORM = MSE_NVF4_PTS / MSE_HiF4

#     MXF4 = To_MXF4(x)
#     MSE_MXF4 = np.sum((MXF4 - x)**2)/(M*N)
#     MSE_MXF4_NORM = MSE_MXF4 / MSE_HiF4

#     HiF4_MSE      [NT] = MSE_HiF4
#     NVF4_MSE      [NT] = MSE_NVF4
#     NVF4_PTS_MSE  [NT] = MSE_NVF4_PTS
#     MXF4_MSE      [NT] = MSE_MXF4

#     MSE_NORM_NVF4 [NT] = MSE_NVF4_NORM
#     MSE_NORM_NVF4_PTS [NT] = MSE_NVF4_PTS_NORM
#     MSE_NORM_MXF4 [NT] = MSE_MXF4_NORM

# print('HiF4_MSE = \n', HiF4_MSE)
# print('NVF4_MSE = \n', NVF4_MSE)
# print('NVF4_PTS_MSE = \n', NVF4_PTS_MSE)
# print('MXF4_MSE = \n', MXF4_MSE)

# print('MSE_NORM_NVF4 = \n', MSE_NORM_NVF4)
# print('Mean MSE_NORM_NVF4 = \n', np.mean(MSE_NORM_NVF4[3:17]))

# print('MSE_NORM_NVF4_PTS = \n', MSE_NORM_NVF4_PTS)
# print('Mean MSE_NORM_NVF4_PTS = \n', np.mean(MSE_NORM_NVF4_PTS))

# print('MSE_NORM_MXF4 = \n', MSE_NORM_MXF4)
# print('Mean MSE_NORM_MXF4 = \n', np.mean(MSE_NORM_MXF4),'\n')

# MSE_NORM_HiF4 = np.ones(Tcnt)
# MSE_NORM = np.vstack([MSE_NORM_MXF4.T, MSE_NORM_NVF4.T,  MSE_NORM_NVF4_PTS.T, MSE_NORM_HiF4]).T
# columns = ['MSE_MXFP4: 4 + 0.25 = 4.25-bit/value', 
#            'MSE_NVFP4: 4 + 0.5 = 4.5-bit/value', 
#            'MSE_NVFP4_PTS: 4 + 0.5 = 4.5-bit/value', 
#            'MSE_HiF4: 4 + 0.5 = 4.5-bit/value']
# idx = range(Tcnt)

# df = pd.DataFrame(MSE_NORM, idx, columns)
# df.plot(
#     linestyle = '-',
#     linewidth = 2,
#     marker = 'o',
#     markersize = 6,
#     markeredgecolor = 'black',
#     markerfacecolor = 'red')

# plt.title('Mean Squared Error Normalized to HiF4', fontsize=14)
# plt.legend(loc='best', fontsize=12)
# plt.xlabel('Gaussian Distribution: mu = 0, sigma = 0.01 * 2 ** X_axis', fontsize=12)
# plt.ylabel('Normalized MSE', fontsize=12)
# plt.show()

# ######--Decimal HiF4--######
# M = 2048
# N = 2048

# sigma = 1.75
# mu = sigma*0.0
# bd = mu
# x = np.random.normal(mu, sigma, (M, N)) + np.random.uniform(-bd, bd, (M, N))

# HiF4 = To_HiFX(x)[0]
# NVF4 = To_NVF4(x)

# # None Zero Rate
# NZR_HiF4 = 1 - np.mean(HiF4 == 0)
# NZR_NVF4 = 1 - np.mean(NVF4 == 0)

# print('None Zero Rate of HiF4 = ', NZR_HiF4*100,'\n')
# print('None Zero Rate of NVF4 = ', NZR_NVF4*100,'\n')


# # #####--Binary HiF4--######

# # HiF4.tofile('input_HiF4_data.bin')

# # # S1P2 Binary Values: 64 B4 --> 
# # U4 = E1M2*4
# # U4 = U4.astype(np.uint8)                     # uint8 expresses E1M2
# # S = np.uint8(Sign == -1) << 3                # Binary Sign: 0 for non-positive; 1 for negative    
# # U4 = U4 | S                                  # SE1M2, Sign Maginitude: Add Sign to E1M2
# # B8_S1P2 = np.zeros((M//2, N), dtype=np.uint8)     # Combine adjacent SE1M2 to form uint8

# # for i in range(0, M//2):                     # Column-wise concatenation: 64 B4 --> 32 B8
# #     for j in range(0, N):
# #         B8_S1P2[i, j] = (U4[2*i, j] << 4) | U4[2*i + 1, j]

# # B8_S1P2_bin = np.array([[format(val, '08b') for val in row] for row in B8_S1P2])
# # print('B8_S1P2 8-bit binary representation = \n', B8_S1P2_bin)

# # B8_S1P2.tofile('input_B8_S1P2_data.bin')

# # # E8_1 + E1_8 --> B16    
# # E8_1 = np.uint16(GE8_1 + 127) << 8     # 8 MSBs of B16, EXP bias = 127

# # E1_8 = np.zeros((M//64, N), dtype=np.uint16)
# # GE1_8 = np.uint16(GE1_8)
# # for i in range(M//64):
# #     for j in range(N):
# #         E1_8[i, j] = (GE1_8[i*8 + 0,j] << 7) | \
# #                      (GE1_8[i*8 + 1,j] << 6) | \
# #                      (GE1_8[i*8 + 2,j] << 5) | \
# #                      (GE1_8[i*8 + 3,j] << 4) | \
# #                      (GE1_8[i*8 + 4,j] << 3) | \
# #                      (GE1_8[i*8 + 5,j] << 2) | \
# #                      (GE1_8[i*8 + 6,j] << 1) | \
# #                      (GE1_8[i*8 + 7,j] << 0)

# # B16_E12 = E8_1 | E1_8

# # B16_E12_bin = np.array([[format(val, '016b') for val in row] for row in B16_E12])
# # print('B16_E12 16-bit binary representation = \n', B16_E12_bin)

# # B16_E12.tofile('input_B16_E12_data.bin')

# # # E1_16 --> B16
# # B16_E3 = np.zeros((M//64, N), dtype=np.uint16)
# # GE1_16 = np.uint16(GE1_16)
# # for i in range(M//64):
# #     for j in range(N):
# #         B16_E3[i, j] = (GE1_16[i*16 + 0, j] << 15) | \
# #                         (GE1_16[i*16 + 1, j] << 14) | \
# #                         (GE1_16[i*16 + 2, j] << 13) | \
# #                         (GE1_16[i*16 + 3, j] << 12) | \
# #                         (GE1_16[i*16 + 4, j] << 11) | \
# #                         (GE1_16[i*16 + 5, j] << 10) | \
# #                         (GE1_16[i*16 + 6, j] << 9 ) | \
# #                         (GE1_16[i*16 + 7, j] << 8 ) | \
# #                         (GE1_16[i*16 + 8, j] << 7 ) | \
# #                         (GE1_16[i*16 + 9, j] << 6 ) | \
# #                         (GE1_16[i*16 + 10,j] << 5 ) | \
# #                         (GE1_16[i*16 + 11,j] << 4 ) | \
# #                         (GE1_16[i*16 + 12,j] << 3 ) | \
# #                         (GE1_16[i*16 + 13,j] << 2 ) | \
# #                         (GE1_16[i*16 + 14,j] << 1 ) | \
# #                         (GE1_16[i*16 + 15,j] << 0 ) 

# # B16_E3_bin = np.array([[format(val, '016b') for val in row] for row in B16_E3])
# # print('B16_E3 16-bit binary representation = \n', B16_E3_bin)

# # B16_E3.tofile('input_B16_E3_data.bin')