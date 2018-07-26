import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from scipy import interpolate
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
Physical parameters
"""
# Available Bandwidth
W = 2*1.024e6
# Doppler Frequency
f_D = 100
# Noise power spectral density
No = 0.00001
# Number of multipath components
IsiDuration = 8
# This flag is used to set the CIR c to a fixed value.
cFlag = False
# Signal-to-noise ratio
#EbNoDB = np.arange(0, 30+1, 5).astype(np.int32)

EbNoDB = np.arange(25, 30+1, 5).astype(np.int32)
'''
MIMO Parameters
'''
N_t = 2
N_r = 2

'''
Design Parameters
'''
# Number of Subcarriers
N = 512
# Subcarrier spacing of OFDM signals
Subcarrier_Spacing = W/N
# Data symbols QAM Modulation Order
m = 4
# Pilot Symbols Modulation Order
m_pilot = 4
# Number of OFDM symbols to simulate for the BER curve
NumOfdmSymbols = 4000
# Total power available for allocation to the subcarriers
Ptotal = 10**(EbNoDB/10)*No*N

'''
Power Amplifier
'''
p_smooth = 1
ClipLeveldB = 3

'''
Secondary Parameters
'''
# OFDM Symbol Duration
T_OFDM = N/W
# OFDM symbol duration including the cyclic prefix.
T_OFDM_Total = (N + IsiDuration -1)/W
# Sampling Period
T_s = 1/W
# Channel Coherence Time
tau_c = 0.5/f_D
# Coherence time in terms of OFDM symbols
L = math.floor(tau_c/T_OFDM_Total)
# Equal power distribution over all subcarriers
Pi = Ptotal/N
# Number of bits for OFDM symbol
NumBitsPerSymbol = m*N
# The normalized signal constellation for data symbols
Const = HelpFunc.UnitQamConstellation(m)
# The normalized signal constellation for pilot symbols
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)

# This variable used for bit-symbol mapping
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
# Number of cyclic prefix samples
CyclicPrefixLen = IsiDuration - 1

'''
Initializations
'''
# Generate a one-sided exponential channel power profile, and normalize total power to 1
temp = CyclicPrefixLen/9 # This line guarantees that the last CIR tap has less power than 0.01 of the first path.
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)


'''
ESN Parameters
'''
# This is the variance of the time-domain
var_x = np.float_power(10, (EbNoDB/10))*No*N
# channel input sequence
nInputUnits = N_t*2
nOutputUnits = N_t*2
# This is the number of neurons in the reservoir. We set this value as a
# function of the time-domain channel input length.
nInternalUnits = 64
inputScaler = 0.005

inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.0000005*np.ones(N_t*2)
spectralRadius = 0.9

# during training.
# Secondary parameters
teacherShift = np.zeros(N_t*2) # No need to introduce a teacher shift
feedbackScaling = feedbackScaler*np.ones(N_t*2)

# Min_Delay and Max_Delay ar the min and max output delays considered in
# training the esn. When the DelayFlag is set more delay quadruplets are
# considered for training, which slows down the script.
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2;
DelayFlag = 0
ESN_train_input = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_train_teacher = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_test_input = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_test_output = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]


'''
Simulation
'''
# The BER and the NMSE matrices to store the simulation results.
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))
c =  [[None] * N_t for i in range(N_r)] # This cell array will store all channel impulse
# responses (CIRs) from the transmit antennas to the receive antennas.
Ci = [[None] * N_t for i in range(N_r)] # This cell array will store the channel frequency
# responses from the transmit antennas to the receive antennas.
Ci_LS  = [[None] * N_t for i in range(N_r)]
Ci_MMSE  = [[None] * N_t for i in range(N_r)]
Ci_LS_Pilots  = [[None] * N_t for i in range(N_r)]
# This is the 1/SNR constant that scales the identity matrix in MMSE channel estimation.
MMSEScaler = (No/Pi)
# Construct the time-domain channel correlation matrix
R_h = np.zeros((IsiDuration, IsiDuration))
for ii in range(IsiDuration):
    R_h[ii, ii] = IsiMagnitude[ii]


for jj in range(len(EbNoDB)):
    print('EbNoDB = %d' % EbNoDB[jj])
    A_Clip = np.sqrt(var_x[jj])* np.float_power(10, ClipLeveldB/20)

    # The ESN parameters that depend on the current SNR
    inputScaling = inputScaler/(var_x[jj]**0.5)*np.ones(N_t*2)
    inputShift = inputOffset/inputScaler*np.ones(N_t * 2)
    # Reset the accumulated number of bit errors for each new SNR value.
    TotalBerNum_ESN = 0
    TotalBerNum_LS = 0
    TotalBerNum_MMSE = 0
    TotalBerNum_Perfect = 0
    TotalBerDen = 0
    # This is just some random last C
    x_ISI = np.zeros(CyclicPrefixLen).astype('complex128')
    NMSE_count = 0
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler[jj]/(N/2)) + np.eye(IsiDuration)

    for kk in range(1, NumOfdmSymbols+1):
        # Check if the current OFDM symbol is for training, or data transmission.
        if (np.remainder(kk, L) == 1):
            # Randomly generate a channel impulse response.
            for nnn in range(N_r):
                for mmm in range(N_t):
                    # Randomly generate a channel impulse response.
                    c[nnn][mmm] = np.random.normal(size=IsiDuration)/(2**0.5) + 1j * np.random.normal(size=IsiDuration)/(2 ** 0.5)
                    c[nnn][mmm] = c[nnn][mmm]*(IsiMagnitude**0.5)
                    Ci[nnn][mmm] = np.fft.fft( np.append(c[nnn][mmm], np.zeros(N - len(c[nnn][mmm]))) )


            '''
            ISI channel + AWGN
            '''
            # The data symbols are M-ary where M = 2^m. We have N_t independent
            # data streams, both of which use the same modulation scheme.
            TxBits = (np.random.uniform(0, 1, size=(N*m_pilot,N_t)) > 0.5).astype(np.int32)

            X = np.zeros((N, N_t)).astype('complex128')
            x_CP = np.zeros((N+CyclicPrefixLen, N_t)).astype('complex128')
            # Modulate the bits with M - ary QAM.
            for ii in range(N):
                for iii in range(N_t):
                    ThisQamIdx = np.matmul(PowersOfTwo[:m_pilot], TxBits[m_pilot * ii + np.arange(m_pilot), iii])
                    X[ii, iii] = ConstPilot[ThisQamIdx[0]]

            #　Create the time - domain signal for each transmit antennas and prepend the cyclic　prefix.
            for iii in range(N_t):
                x_temp = N * np.fft.ifft(X[:, iii])
                x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_CP[:, iii] = x_CP[:, iii] * (Pi[jj]**0.5)
            Y = np.zeros((N, N_r)).astype('complex128')
            y_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
            noise = np.zeros(y_CP.shape).astype('complex128')

            for nnn in range(N_r):
                    # Superimpose all transmitted and filtered streams to form
                    # the received time domain signal in each receive antenna.
                    for mmm in range(N_t):
                        y_CP[:,nnn] = y_CP[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP[:,mmm])
                    # Add noise to the signal at each receive antenna.
                    noise[:,nnn] = math.sqrt(y_CP.shape[0]*No/2) * np.matmul(np.random.normal(size=(y_CP.shape[0], 2)),
                                                              np.array([[1], [1j]])).reshape(-1)
                    y_CP[:,nnn] = y_CP[:,nnn] + noise[:,nnn]
                    # Get the frequency domain samples at each receive antenna.
                    Y[:, nnn] = 1 / N * np.fft.fft(y_CP[IsiDuration-1:len(y_CP),nnn])


            # Insert zeros at the subcarriers that are pilot locations for the other antenna.
            X_LS = X
            X_LS[np.arange(1, len(X_LS), 2), 0] = 0
            X_LS[np.arange(0, len(X_LS), 2), 1] = 0
            x_LS_CP = np.zeros(x_CP.shape).astype('complex128')
            for iii in range(N_t):
                x_temp = N*np.fft.ifft(X_LS[:,iii])
                x_LS_CP[:,iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_LS_CP[:,iii] = x_LS_CP[:,iii] * (Pi[jj] ** 0.5)
            Y_LS = np.zeros((N,N_r)).astype('complex128')
            y_LS_CP = np.zeros((N+CyclicPrefixLen, N_r)).astype('complex128')
            for nnn in range(N_r):
                for mmm in range(N_t):
                    y_LS_CP[:,nnn] = y_LS_CP[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_LS_CP[:, mmm])

                y_LS_CP[:, nnn] = y_LS_CP[:,nnn] + noise[:,nnn]
                # Get the frequency domain samples at each receive antenna.
                Y_LS[:,nnn] = 1/N*np.fft.fft(y_LS_CP[IsiDuration-1:,nnn])
            Y_LS = Y_LS/(Pi[jj]**0.5)
            '''
            LS Channel Estimation
            '''
            for nnn in range(N_r):
                for mmm in range(N_t):
                    Ci_LS_Pilots[nnn][mmm] = Y_LS[np.arange(mmm, len(Y_LS), 2), nnn]/ X_LS[np.arange(mmm, len(X_LS), 2), mmm]
                    # MMSE Channel Estimation
                    #  H_MMSE = MMSE_bold_FD * H_LS
                    c_LS = np.fft.ifft(Ci_LS_Pilots[nnn][mmm])
                    # Get time - domain LS estimates at the known CIR locations.
                    c_LS = np.delete(c_LS, np.arange(IsiDuration, len(c_LS)))
                    # Get the MMSE estimate from the LS estimate
                    c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS)
                    Ci_MMSE[nnn][mmm] = np.fft.fft(np.append(c_MMSE, np.zeros(N-IsiDuration)))
                    if (mmm == 0):
                        tmpf = interpolate.interp1d(np.append(np.arange(mmm, N-1, N_t), N-1),
                                 np.append(Ci_LS_Pilots[nnn][mmm], Ci_LS_Pilots[nnn][mmm][-1]))
                        Ci_LS[nnn][mmm] = tmpf(np.arange(N))
                    else:
                        tmpf = interpolate.interp1d(np.append(0, np.arange(mmm, N, N_t)),
                                 np.append(Ci_LS_Pilots[nnn][mmm][0], Ci_LS_Pilots[nnn][mmm]))
                        Ci_LS[nnn][mmm] = tmpf(np.arange(N))
            '''
            ESN Receiver
            '''
            # Pass the time-domain samples through the nonlinear PA
            x_CP_NLD = x_CP/( ( 1 + (np.absolute(x_CP)/A_Clip)**(2*p_smooth) )**(1/(2*p_smooth)) )

            y_CP_NLD = np.zeros((N+CyclicPrefixLen,N_r)).astype('complex128')
            for nnn in range(N_r):
                    # Superimpose all transmitted and filtered streams to form
                    # the received time domain signal in each receive antenna.
                    for mmm in range(N_t):
                        y_CP_NLD[:, nnn] = y_CP_NLD[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
                    y_CP_NLD[:, nnn] = y_CP_NLD[:,nnn] + noise[:,nnn]
            # Set-up the ESN
            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                                      spectral_radius=spectralRadius, sparsity= 1 - min(0.2*nInternalUnits, 1),
                                      input_shift=inputShift, input_scaling=inputScaling,
                                      teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                                      feedback_scaling=feedbackScaling)
            # Train the ESN
            [ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN] = \
            HelpFunc.trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP_NLD, x_CP)

            # Data For AFRL
            #ESN_train_input[kk, jj] = ESN_input
            #ESN_train_teacher[kk, jj] = ESN_output

            NMSE_ESN_Training[jj] = NMSE_ESN + NMSE_ESN_Training[jj]

        else:
            TxBits = (np.random.uniform(0, 1, size=(N*m, N_t)) > 0.5).astype(np.int32)

            X = np.zeros((N, N_t)).astype('complex128')
            x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
            # Modulate the bits with M - ary QAM.
            for ii in range(N):
                for iii in range(N_t):
                    ThisQamIdx = np.matmul(PowersOfTwo[:m], TxBits[m * ii + np.arange(m), iii])
                    X[ii, iii] = Const[ThisQamIdx[0]]

            for iii in range(N_t):
                x_temp = N * np.fft.ifft(X[:, iii])
                x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

            # nonlinear Amplifier

            x_CP_NLD = x_CP / ((1 + (np.absolute(x_CP)/A_Clip) ** (2*p_smooth)) ** (1/(2*p_smooth)))


            Y_NLD = np.zeros((N, N_r)).astype('complex128')
            y_CP_NLD = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')

            for nnn in range(N_r):
                # Superimpose all transmitted and filtered streams to form
                # the received time domain signal in each receive antenna.
                for mmm in range(N_t):
                    y_CP_NLD[:, nnn] = y_CP_NLD[:, nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
# add noise
                y_CP_NLD[:, nnn] = y_CP_NLD[:, nnn] + math.sqrt(y_CP.shape[0] * No / 2) \
                     * np.matmul(np.random.normal(size=(y_CP.shape[0], 2)), np.array([[1], [1j]])).reshape(-1)
                Y_NLD[:, nnn] = 1 / N * np.fft.fft(y_CP_NLD[IsiDuration-1:, nnn])



            # ESN Detector
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
            ESN_output = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))

            ESN_input[:, 0] = np.append(y_CP_NLD[:, 0].real, np.zeros(Delay_Max))
            ESN_input[:, 1] = np.append(y_CP_NLD[:, 0].imag, np.zeros(Delay_Max))
            ESN_input[:, 2] = np.append(y_CP_NLD[:, 1].real, np.zeros(Delay_Max))
            ESN_input[:, 3] = np.append(y_CP_NLD[:, 1].imag, np.zeros(Delay_Max))

            # Get the ESN output corresponding to the training input
            # Train the ESN
            nForgetPoints = Delay_Min + CyclicPrefixLen
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, nForgetPoints, continuation=False)

            x_hat_ESN_0 = x_hat_ESN_temp[Delay[0] - Delay_Min: Delay[0] - Delay_Min + N + 1, 0] \
                            + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min: Delay[1] - Delay_Min + N + 1, 1]
            x_hat_ESN_1 = x_hat_ESN_temp[Delay[2] - Delay_Min: Delay[2] - Delay_Min + N + 1, 2] \
                            + 1j * x_hat_ESN_temp[Delay[3] - Delay_Min: Delay[3] - Delay_Min + N + 1, 3]



            x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
            x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)
            x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))

            x = x_CP[IsiDuration - 1:, :]

            NMSE_ESN_Testing[jj] = NMSE_ESN_Testing[jj] \
                + np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0], axis=0) ** 2 / np.linalg.norm(x[:, 0], axis=0) ** 2 \
                + np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1], axis=0) ** 2 / np.linalg.norm(x[:, 1], axis=0) ** 2

            NMSE_count = NMSE_count + 1
            X_hat_ESN = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_ESN[:, ii] = 1 / N * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi[jj])
            # Data For AFRL
            #ESN_test_input[kk, jj] = ESN_input
            #ESN_test_output[kk, jj] = x_hat_ESN

            '''
            Channel Estimation
            '''
            H_temp = np.zeros((N_r, N_t)).astype('complex128')
            H_temp_LS = np.zeros((N_r, N_t)).astype('complex128')
            H_temp_MMSE = np.zeros((N_r, N_t)).astype('complex128')
            X_hat_Perfect = np.zeros(X.shape).astype('complex128')
            X_hat_LS = np.zeros(X.shape).astype('complex128')
            X_hat_MMSE = np.zeros(X.shape).astype('complex128')

            for ii in range(N):
                Y_temp = np.transpose(Y_NLD[ii,:])
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        H_temp[nnn, mmm] = Ci[nnn][mmm][ii]
                        H_temp_LS[nnn, mmm] = Ci_LS[nnn][mmm][ii]
                        H_temp_MMSE[nnn, mmm] = Ci_MMSE[nnn][mmm][ii]

                X_hat_Perfect[ii,:] = np.linalg.solve(H_temp, Y_temp) / math.sqrt(Pi[jj])
                X_hat_LS[ii,:] = np.linalg.solve(H_temp_LS, Y_temp) / math.sqrt(Pi[jj])
                X_hat_MMSE[ii,:] = np.linalg.solve(H_temp_MMSE, Y_temp) / math.sqrt(Pi[jj])

            RxBits_ESN = np.zeros(TxBits.shape)
            RxBits_LS = np.zeros(TxBits.shape)
            RxBits_MMSE = np.zeros(TxBits.shape)
            RxBits_Perfect = np.zeros(TxBits.shape)
            # Loop through the subcarriers and detect the QAM symbols and bits.
            for ii in range(N):
                for iii in range(N_t):
                    # Bit and symbol detection with the "exact" equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_Perfect[ii,iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_Perfect[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with ESN Receiver
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_ESN[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_ESN[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with LS equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_LS[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_LS[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with MMSE equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_MMSE[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_MMSE[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

            # Accumulate the bit errors for all three receivers
            TotalBerNum_ESN = TotalBerNum_ESN + np.sum(TxBits != RxBits_ESN)
            TotalBerNum_LS = TotalBerNum_LS + np.sum(TxBits != RxBits_LS)
            TotalBerNum_MMSE = TotalBerNum_MMSE + np.sum(TxBits != RxBits_MMSE)
            TotalBerNum_Perfect = TotalBerNum_Perfect + np.sum(TxBits != RxBits_Perfect)
            TotalBerDen = TotalBerDen + NumBitsPerSymbol

    # Compute and store the bit error rate(BER) values for the current signal to noise ratio.
    BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen
    BER_LS[jj] = TotalBerNum_LS / TotalBerDen
    BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen
    BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen

NMSE_ESN_Testing = NMSE_ESN_Testing / NMSE_count
NMSE_ESN_Training = NMSE_ESN_Training / (NumOfdmSymbols - NMSE_count)

# Plot the BER of all three approaches.


BERvsEBNo = {
"EBN0":EbNoDB,
"BER": BER_ESN}

f = open('./BERvsEBNo_esn.pkl','wb')

pickle.dump(BERvsEBNo,f)

f.close()


plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS', linewidth=1.5)
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE', linewidth=1.5)
plt.legend()
plt.grid(True)
plt.title('64 Neurons in the Reservoir')
plt.xlabel('Signal-to-Noise Ratio[dB]')
plt.ylabel('Bit Error Rate')
plt.show()


