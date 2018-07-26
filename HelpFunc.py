import math
import numpy as np

class HelpFunc():

    def UnitQamConstellation(Bi):
        ''' '''
        '''
        When Bi is even, then M has the following properties:
            * it has an integer square root
            * the square root is an even number
        When Bi is odd, we want to round its square root up to the next even number,
        in order to find the smallest even-sided square that will hold our points.
        '''
        EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2

        '''
        We need a PAM-type alphabet based on this even square root
        '''
        PamM = EvenSquareRoot

        '''
        Now, make the square QAM constellation using the basic PAM order:
            * Start with the basic M-ary PAM constellation
            * Make an M-by-M matrix where each row is the basic M-ary PAM constellation
            * Make a copy of the M-by-M matrix, and then transpose the copy
            * Multiply the first matrix by 1, and the second matrix by j, then add
        '''
        PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32)
        PamConstellation = np.reshape(PamConstellation, (1, -1))
        SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
        C = SquareMatrix + 1j * (SquareMatrix.T)

        C_tmp = np.zeros(C.shape[0]*C.shape[1]).astype('complex128')
        for i in range(C.shape[1]):
            for j in range(C.shape[0]):
                C_tmp[i*C.shape[0] + j] = C[j][i]
        C = C_tmp
        return C / math.sqrt(np.mean(abs(C) ** 2))

    def ComputeChannelCorrMatrix(IsiMagnitude):
        '''
        % This function takes the power (expected value of magnitude square) of
        % each multipath component, and computes the channel correlation matrix.
        % The channel correlation matrix (in frequency) has the conjugate symmetry
        % across the main diagonal, so it can be populated from the first row.
        '''
        N = len(IsiMagnitude)
        # Construct the Channel Correlation Matrix
        # First compute the frequency correlation vector
        r_f_bold = np.fft.fft(IsiMagnitude)
        # The(n, m) the element of the channel matrix is r_f_bold(n - m),
        # if (n - m) < 0, then we use the relationship: r_f_bold(n - m) = conj(r_f_bold(m - n)).
        # Below is one way to populate the channel correlation matrix denoted by R_f.
        r_f_bold_prime = r_f_bold[1:]
        r_f_bold_prime = r_f_bold_prime[::-1]
        r_f_bold_conj = np.conjugate(r_f_bold)
        r_f_bold_ext = np.append(r_f_bold_prime, r_f_bold_conj)
        R_f = np.zeros((N, N)).astype('complex128')
        for k in range(0, N):
            R_f[N - k - 1, :] = r_f_bold_ext[k: N + k]
        return R_f

    def trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP):
        # This function trains the ESN for a 2x2 MIMO system for various values of
        # the output delays at the 4 output nodes denoted by d_1, d_2, d_3, d_4.
        # For each delay quadruple ut computes the NMSE. Then it picks the delay
        # quadruple with the lowest NMSE, and trains the esn with the chosen
        # delays. First a delay look-up-table with different delays is generates.
        # If the flag is set we consider a larger number of delays quadruples. If
        # This approach reduces the training time greatly.

        if (DelayFlag):
            Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay) ** 2, 4)).astype('int32')
            count = 0
            temp = np.zeros(Delay_LUT.shape[0], 1)
            for ii in range(Min_Delay, Max_Delay + 1):
                for jj in range(Min_Delay, Max_Delay + 1):
                    for kk in range(Min_Delay, Max_Delay + 1):
                        for ll in range(Min_Delay, Max_Delay + 1):
                            Delay_LUT[count, :] = np.array([ii, jj, kk, ll])
                            if (abs(ii - jj) > 2):
                                temp[count] = 1
                            elif (abs(kk - ll) > 2):
                                temp[count] = 1
                            elif (abs(ii - kk) > 2):
                                temp[count] = 1
                            elif (abs(ii - ll) > 2):
                                temp[count] = 1
                            elif (abs(jj - kk) > 2):
                                temp[count] = 1
                            elif (abs(jj - ll) > 2):
                                temp[count] = 1
                            count = count + 1

            np.delete(Delay_LUT, np.argwhere(temp > 0), 0)

        else:
            Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay), 4)).astype('int32')
            for jjjj in range(0, Max_Delay + 1):
                Delay_LUT[jjjj, :] = (jjjj) * np.ones(4)

        # Compute the max and the min of ach dealy row.
        Delay_Max = np.amax(Delay_LUT, axis=1)
        Delay_Min = np.amin(Delay_LUT, axis=1)

        # Train the esn with different delays and store the training NMSE.
        NMSE_ESN_Training = np.zeros(Delay_LUT.shape[0])
        for jjj in range(Delay_LUT.shape[0]):
            Curr_Delay = Delay_LUT[jjj, :]

            ESN_input = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))
            ESN_output = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))

            # The ESN input
            ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[jjj]))

            # The ESN output
            ESN_output[Curr_Delay[0]: (Curr_Delay[0] + N + CyclicPrefixLen), 0] = x_CP[:, 0].real
            ESN_output[Curr_Delay[1]: (Curr_Delay[1] + N + CyclicPrefixLen), 1] = x_CP[:, 0].imag
            ESN_output[Curr_Delay[2]: (Curr_Delay[2] + N + CyclicPrefixLen), 2] = x_CP[:, 1].real
            ESN_output[Curr_Delay[3]: (Curr_Delay[3] + N + CyclicPrefixLen), 3] = x_CP[:, 1].imag

            # Train the ESN
            nForgetPoints = Delay_Min[jjj] + CyclicPrefixLen
            esn.fit(ESN_input, ESN_output, nForgetPoints)

            # Get the ESN output corresponding to the training input
            x_hat_ESN_temp = esn.predict(ESN_input, nForgetPoints, continuation=False)

            # Put the real and the imaginary parts of each transmitted stream together
            x_hat_ESN_0 = x_hat_ESN_temp[Curr_Delay[0] - Delay_Min[jjj]: Curr_Delay[0] - Delay_Min[jjj] + N + 1, 0] \
                          + 1j * x_hat_ESN_temp[Curr_Delay[1] - Delay_Min[jjj]: Curr_Delay[1] - Delay_Min[jjj] + N + 1,
                                 1]

            x_hat_ESN_1 = x_hat_ESN_temp[Curr_Delay[2] - Delay_Min[jjj]: Curr_Delay[2] - Delay_Min[jjj] + N + 1, 2] \
                          + 1j * x_hat_ESN_temp[Curr_Delay[3] - Delay_Min[jjj]: Curr_Delay[3] - Delay_Min[jjj] + N + 1,
                                 3]

            x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
            x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)


            x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))

            # Compute the training NMSE
            x = x_CP[IsiDuration - 1:, :]

            NMSE_ESN_Training[jjj] = \
                np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0], axis=0) ** 2 / np.linalg.norm(x[:, 0], axis=0) ** 2 \
              + np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1], axis=0) ** 2 / np.linalg.norm(x[:, 1], axis=0) ** 2

        # Find the delay row that minimizes the NMSE
        Delay_Idx = np.argmin(NMSE_ESN_Training)

        Delay_Idx=3

        print(NMSE_ESN_Training)
        NMSE_ESN = np.amin(NMSE_ESN_Training)
        Delay = Delay_LUT[Delay_Idx, :]

        # Train the esn with the optimal delay quadruple.
        ESN_input = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))
        ESN_output = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))

        # The ESN input
        ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[Delay_Idx]))

        # The ESN output
        ESN_output[Delay[0]: (Delay[0] + N + CyclicPrefixLen), 0] = x_CP[:, 0].real
        ESN_output[Delay[1]: (Delay[1] + N + CyclicPrefixLen), 1] = x_CP[:, 0].imag
        ESN_output[Delay[2]: (Delay[2] + N + CyclicPrefixLen), 2] = x_CP[:, 1].real
        ESN_output[Delay[3]: (Delay[3] + N + CyclicPrefixLen), 3] = x_CP[:, 1].imag

        nForgetPoints = Delay_Min[Delay_Idx] + CyclicPrefixLen
        esn.fit(ESN_input, ESN_output, nForgetPoints)

        Delay_Minn = Delay_Min[Delay_Idx]
        Delay_Maxx = Delay_Max[Delay_Idx]

        return [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Minn,  Delay_Maxx, nForgetPoints, NMSE_ESN]

