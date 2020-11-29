%%%%%%%%%%%%%%%%%%%%%%% SCRIPT DESCRIPTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%% Physical Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = 2.*1.024e6;           % Available Bandwidth
f_D = 100;             % Doppler Frequency
No        = 0.00001;  % Noise power spectral density
IsiDuration  = 8;     % Number of multipath components  
EbNoDB = 2:4:25;          % Signal-to-noise ratio
%%%%%%%%%%%%%%%%%%%%%% MIMO Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_t = 2; % Number of Transmit Antennas
N_r = 2; % Number of Receive Antennas
%%%%%%%%%%%%%%%%%%%%%% Design Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N  = 512;            % Number of Subcarriers
Subcarrier_Spacing = W/N; % Subcarrier spacing of OFDM signals. 
m = 2;                % Data symbols QAM Modulation Order
m_pilot = 2;          % Pilot Symbols Modulation Order
NumOfdmSymbols = 1000; % Number of OFDM symbols to simulate for the BER curve 
Ptotal    = 10.^(EbNoDB./10).*No.*N;  % Total power available for allocation to the subcarriers
%%%%%%%%%%%%%%%%%%%%% Secondary Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_OFDM = N/W;  % OFDM Symbol Duration
T_OFDM_Total = (N+IsiDuration -1)/W; % OFDM symbol duration including the cyclic prefix.
T_s = 1/W;     % Sampling Period 
tau_c = 0.5/f_D; % Channel Coherence Time
L = floor(tau_c/T_OFDM_Total); % Coherence time in terms of OFDM symbols
Pi = Ptotal./N; % Equal power distribution over all subcarriers
NumBitsPerSymbol = m*N; % Number of bits for OFDM symbol
Const = UnitQamConstellation(m); % The normalized signal constellation for data symbols
ConstPilot = UnitQamConstellation(m_pilot); % The normalized signal constellation for pilot symbols
PowersOfTwo     = 2.^[0:m-1]; % This variable used for bit-symbol mapping.
CyclicPrefixLen = IsiDuration - 1; % Number of cyclic prefix samples. 
%%%%%%%%%%%%%%%%%%%%% Initializations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the random number generators.
rand('state',sum(100*clock));
randn('state',sum(100*clock));
% Generate a one-sided exponential channel power profile, and normalize total power to 1.
temp = CyclicPrefixLen/9; % This line guarantees that the last CIR tap has 
% less power than 0.01 of the first path. 
IsiMagnitude = exp(-(0:CyclicPrefixLen)./temp)';
IsiMagnitude = IsiMagnitude./sum(IsiMagnitude);
%%%%%%%%%%%%%%%%%%%% ESN Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
var_x = 10.^(EbNoDB./10).*No.*N; % This is the variance of the time-domain 
% channel input sequence. 
nInputUnits = N_t*2; 
nOutputUnits = N_t*2;
% This is the number of neurons in the reservoir. We set this value as a
% function of the time-domain channel input length. 

% nInternalUnits = N/8;
% inputScaler = 0.35;
% inputOffset = 0.75;
% feedbackScaler = 0.4;
% teacherScaling = 0.0005.*ones(N_t*2,1);
% spectralRadius = 0.80;

nInternalUnits = N/8;
inputScaler = 0.00005;
inputOffset = 0.00000;
feedbackScaler = 0.000000;
teacherScaling = 0.000005.*ones(N_t*2,1);
spectralRadius = 0.80;

% Secondary parameters
teacherShift =  zeros(N_t*2,1); % No need to introduce a teacher shift. 
feedbackScaling =  feedbackScaler.*ones(N_t*2,1);
% Min_Delay and Max_Delay ar the min and max output delays considered in
% training the esn. When the DelayFlag is set more delay quadruplets are
% considered for training, which slows down the script. 
Min_Delay = 0;
Max_Delay = ceil(IsiDuration/2)+2;
DelayFlag = 0;
%%%%%%%%%%%%%%%%%%%%%%% Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The BER and the NMSE matrices to store the simulation results. 
BER_ESN = zeros(length(EbNoDB),1);
BER_Perfect = zeros(length(EbNoDB),1);
BER_LS = zeros(length(EbNoDB),1); 
BER_MMSE = zeros(length(EbNoDB),1);

NMSE_ESN_Testing = zeros(length(EbNoDB),1);
NMSE_ESN_Training = zeros(length(EbNoDB),1);
c = cell(N_r,N_t); % This cell array will store all channel impulse 
% responses (CIRs) from the transmit antennas to the receive antennas. 
Ci = cell(N_r,N_t); % This cell array will store the channel frequency
% responses from the transmit antennas to the receive antennas.
Ci_LS  = cell(N_r,N_t);
Ci_MMSE  = cell(N_r,N_t);
Ci_LS_Pilots  = cell(N_r,N_t);
% This is the 1/SNR constant that scales the identity matrix in MMSE
% channel estimation.
MMSEScaler = (No./Pi);
% Construct the time-domain channel correlation matrix
R_h = zeros(IsiDuration);
for ii =1:IsiDuration
    R_h(ii,ii) = IsiMagnitude(ii);
end

for jj = 1:length(EbNoDB)
    % The ESN parameters that depend on the current SNR
    inputScaling =  inputScaler./sqrt(var_x(jj)).*ones(N_t*2,1);
    inputShift = inputOffset./inputScaler.*ones(N_t*2,1);
    % Reset the accumulated number of bit errors for each new SNR value.
    TotalBerNum_ESN = 0;
    TotalBerNum_LS = 0;
    TotalBerNum_MMSE = 0;
    TotalBerNum_Perfect = 0;
    TotalBerDen = 0;
    % This is just some random last C
    x_ISI = zeros(CyclicPrefixLen,1);
    NMSE_count = 0;
    MMSE_bold_TD = inv(R_h).*MMSEScaler(jj)./(N/2) + eye(IsiDuration);
    for kk = 1:NumOfdmSymbols 
        % Check if the current OFDM symbol is for training, or data transmission.
        if(mod(kk,L) == 1)
            % Randomly generate a channel impulse response. 
            for nnn = 1:N_r
                for mmm = 1:N_t
                    % Randomly generate a channel impulse response. 
                    c{nnn}{mmm} = randn(IsiDuration,1)./sqrt(2) + 1i*randn(IsiDuration,1)./sqrt(2); 
                    c{nnn}{mmm} = c{nnn}{mmm}.*sqrt(IsiMagnitude); 
                    Ci{nnn}{mmm} = fft([c{nnn}{mmm};zeros(N-length(c{nnn}{mmm}),1)]);
                end
            end
            %%%%%%%%%%%%%%%%%%% ISI channel + AWGN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % The data symbols are M-ary where M = 2^m. We have N_t independent
            % data streams, both of which use the same modulation scheme. 
            TxBits = 0.5 < rand(N*m_pilot,N_t);     
            X = zeros(N,N_t); 
            x_CP = zeros(N+CyclicPrefixLen,N_t);
            % Modulate the bits with M-ary QAM. 
            for ii = 1:N
                for iii = 1:N_t
                    ThisQamIdx = PowersOfTwo(1:m_pilot)*TxBits(m_pilot*(ii-1) + [1:m_pilot], iii);
                    X(ii,iii)      = ConstPilot(ThisQamIdx+1);
                end
            end
            % Create the time-domain signal for each transmit antennas and 
            % prepend the cyclic prefix.
            for iii = 1:N_t
                x_temp = N*ifft(X(:,iii));
                x_CP(:,iii) = [x_temp(end-CyclicPrefixLen+1:end);x_temp];
                x_CP(:,iii) = x_CP(:,iii).*sqrt(Pi(jj));
            end            
            Y = zeros(N,N_r);
            y_CP = zeros(N+CyclicPrefixLen,N_r);
            noise = zeros(size(y_CP));
            for nnn = 1:N_r
                    % Superimpose all transmitted and filtered streams to form
                    % the received time domain signal in each receive antenna. 
                    for mmm = 1:N_t
                        y_CP(:,nnn) = y_CP(:,nnn) + filter(c{nnn}{mmm},1,x_CP(:,mmm));
                    end
                    % Add noise to the signal at each receive antenna. 
                    noise(:,nnn) = sqrt(size(y_CP,1)*No/2)*randn(size(y_CP,1),2)*[1;j];
                    y_CP(:,nnn) = y_CP(:,nnn) + noise(:,nnn);
                    % Get the frequency domain samples at each receive antenna.
                    Y(:,nnn) = 1/N*fft(y_CP(IsiDuration:end,nnn));
            end
            % Insert zeros at the subcarriers that are pilot locations for
            % the other antenna. 
            X_LS = X;
            X_LS(2:2:end,1) = 0.*X_LS(2:2:end,1);
            X_LS(1:2:end,2) = 0.*X_LS(1:2:end,2);
            x_LS_CP = zeros(size(x_CP));
            for iii = 1:N_t
                x_temp = N*ifft(X_LS(:,iii));
                x_LS_CP(:,iii) = [x_temp(end-CyclicPrefixLen+1:end);x_temp];
                x_LS_CP(:,iii) = x_LS_CP(:,iii).*sqrt(Pi(jj));
            end 
            Y_LS = zeros(N,N_r);
            y_LS_CP = zeros(N+CyclicPrefixLen,N_r);
            for nnn = 1:N_r 
                for mmm = 1:N_t
                    y_LS_CP(:,nnn) = y_LS_CP(:,nnn) + filter(c{nnn}{mmm},1,x_LS_CP(:,mmm));
                end
                y_LS_CP(:,nnn) = y_LS_CP(:,nnn) + noise(:,nnn);
                % Get the frequency domain samples at each receive antenna.
                Y_LS(:,nnn) = 1/N*fft(y_LS_CP(IsiDuration:end,nnn));
            end
            Y_LS = Y_LS./sqrt(Pi(jj));
            %%%%%%%%%%%% LS Channel Estimation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for nnn = 1:N_r
                for mmm = 1:N_t
                    Ci_LS_Pilots{nnn}{mmm} = Y_LS(mmm:2:end,nnn)./X_LS(mmm:2:end,mmm);
                    % MMSE Channel Estimation
                    % H_MMSE = MMSE_bold_FD*H_LS;
                    c_LS = ifft(Ci_LS_Pilots{nnn}{mmm});
                    % Get time-domain LS estimates at the known CIR locations.
                    c_LS(IsiDuration+1:end) = [];
                    % Get the MMSE estimate from the LS estimate
                    c_MMSE = MMSE_bold_TD\c_LS;
                    Ci_MMSE{nnn}{mmm} = fft([c_MMSE; zeros(N-IsiDuration,1)]);
                    if(mmm == 1)
                        Ci_LS{nnn}{mmm} = interp1([(mmm:N_t:N-1)';N],[Ci_LS_Pilots{nnn}{mmm};...
                                Ci_LS_Pilots{nnn}{mmm}(end)],(1:N)','linear');
                    else
                        Ci_LS{nnn}{mmm} = interp1([1;(mmm:N_t:N)'],[Ci_LS_Pilots{nnn}{mmm}(1);...
                                Ci_LS_Pilots{nnn}{mmm}],(1:N)','linear');
                    end
                end
            end
            %%%%%%%%%%%% ESN Channel Equalization %%%%%%%%%%%%%%%%%%%%%%%%%
            % Set-up the ESN
            esn =  generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
                'spectralRadius',spectralRadius ,'inputScaling',inputScaling,'inputShift',inputShift, ...
                'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling',feedbackScaling, ...
                'learningMode', 'offline_singleTimeSeries') ; 
            % Set the spectral radius
            esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;            
            % Train the ESN
            %[trainedEsn, Delay, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN]
            [ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN] = trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, ...
                      CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP);  
            NMSE_ESN_Training(jj) = NMSE_ESN + NMSE_ESN_Training(jj);
        else 
            TxBits = 0.5 < rand(N*m,N_t);     
            X = zeros(N,N_t);
            x_CP = zeros(N+CyclicPrefixLen,N_t);
            % Modulate the bits with M-ary QAM. 
            for ii = 1:N
                for iii = 1:N_t
                    ThisQamIdx = PowersOfTwo(1:m)*TxBits(m*(ii-1) + [1:m], iii);
                    X(ii,iii)      = Const(ThisQamIdx+1);
                end
            end
            for iii = 1:N_t
                x_temp = N*ifft(X(:,iii));
                x_CP(:,iii) = [x_temp(end-CyclicPrefixLen+1:end);x_temp];
                x_CP(:,iii) = x_CP(:,iii).*sqrt(Pi(jj));
            end
            
            Y = zeros(N,N_r);
            y_CP = zeros(N+CyclicPrefixLen,N_r);
            for nnn = 1:N_r 
                    for mmm = 1:N_t
                        y_CP(:,nnn) = y_CP(:,nnn) + filter(c{nnn}{mmm},1,x_CP(:,mmm));
                    end 
                    y_CP(:,nnn) = y_CP(:,nnn) + sqrt(size(y_CP,1)*No/2)*randn(size(y_CP,1),2)*[1;j];
                    Y(:,nnn) = 1/N*fft(y_CP(IsiDuration:end,nnn));
            end
            %%%%%%%%%%%%%%%%% ESN Equalizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ESN_input = zeros(N + Delay_Max + CyclicPrefixLen,N_t*2);
            ESN_output = zeros(N + Delay_Max+ CyclicPrefixLen,N_t*2);

            ESN_input(:,1) = [real(y_CP(:,1)); zeros(Delay_Max,1)];
            ESN_input(:,2) = [imag(y_CP(:,1)); zeros(Delay_Max,1)];
            ESN_input(:,3) = [real(y_CP(:,2)); zeros(Delay_Max,1)];
            ESN_input(:,4) = [imag(y_CP(:,2)); zeros(Delay_Max,1)];
            % Get the ESN output corresponding to the training input
            % Train the ESN
            nForgetPoints = Delay_Min + CyclicPrefixLen;
            x_hat_ESN_temp = test_esn(ESN_input, trainedEsn, nForgetPoints);
            x_hat_ESN(:,1) = x_hat_ESN_temp(Delay(1) - Delay_Min+1: Delay(1) - Delay_Min+N, 1) ...
                + j.*x_hat_ESN_temp(Delay(2) - Delay_Min+1: Delay(2) - Delay_Min+N, 2);
            x_hat_ESN(:,2) = x_hat_ESN_temp(Delay(3) - Delay_Min+1: Delay(3) - Delay_Min+N, 3) ...
                + j.*x_hat_ESN_temp(Delay(4) - Delay_Min+1: Delay(4) - Delay_Min+N, 4);
            
            x = x_CP(IsiDuration:end,:);
            NMSE_ESN_Testing(jj) = NMSE_ESN_Testing(jj) + ((x_hat_ESN(:,1) - x(:,1) )'*(x_hat_ESN(:,1)  - x(:,1) ))./(x(:,1)'*x(:,1) )...
                + ((x_hat_ESN(:,2) - x(:,2) )'*(x_hat_ESN(:,2)  - x(:,2) ))./(x(:,2)'*x(:,2) ); 
            NMSE_count  = NMSE_count + 1;
            X_hat_ESN = zeros(size(X));
            for ii= 1:N_t
                X_hat_ESN(:,ii) = fft(x_hat_ESN(:,ii))./N./sqrt(Pi(jj));
            end
            %%%%%%%%%%%%%%%% Channel Estimation %%%%%%%%%%%%%%%%%%%%%%%%%%%
            H_temp = zeros(N_r,N_t);
            H_temp_LS = zeros(N_r,N_t);
            H_temp_MMSE = zeros(N_r,N_t);
            X_hat_Perfect = zeros(size(X));
            X_hat_LS = zeros(size(X));
            X_hat_MMSE = zeros(size(X));
            for ii = 1:N 
                Y_temp = (Y(ii,:)).';
                for nnn = 1: N_r
                    for mmm = 1:N_t
                        H_temp(nnn,mmm) = Ci{nnn}{mmm}(ii);
                        H_temp_LS(nnn,mmm) = Ci_LS{nnn}{mmm}(ii);
                        H_temp_MMSE(nnn,mmm) = Ci_MMSE{nnn}{mmm}(ii);
                    end
                end   
                X_hat_Perfect(ii,:) =H_temp\Y_temp./sqrt(Pi(jj));
                X_hat_LS(ii,:) = H_temp_LS\Y_temp/sqrt(Pi(jj));
                X_hat_MMSE(ii,:) = H_temp_MMSE\Y_temp/sqrt(Pi(jj));
            end
            RxBits_ESN = zeros(size(TxBits));
            RxBits_LS = zeros(size(TxBits));
            RxBits_MMSE = zeros(size(TxBits));
            RxBits_Perfect = zeros(size(TxBits));
            % Loop through the subcarriers and detect the QAM symbols and bits.
            for ii = 1:N
                for iii = 1:N_t
                    % Bit and symbol detection with the "exact" equalizer
                    [junk,ThisQamIdx] = min(abs(Const - X_hat_Perfect(ii,iii)));
                    ThisBits = abs(dec2bin(ThisQamIdx-1,m) - '0');
                    RxBits_Perfect((m*(ii-1)+1):m*ii,iii) = ThisBits(end:-1:1);
                    % Bit and symbol detection with ESN equalizer
                    [junk,ThisQamIdx] = min(abs(Const - X_hat_ESN(ii,iii)));
                    ThisBits = abs(dec2bin(ThisQamIdx-1,m) - '0');
                    RxBits_ESN((m*(ii-1)+1):m*ii,iii) = ThisBits(end:-1:1);
                    % Bit and symbol detection with LS equalizer
                    [junk,ThisQamIdx] = min(abs(Const - X_hat_LS(ii,iii)));
                    ThisBits = abs(dec2bin(ThisQamIdx-1,m) - '0');
                    RxBits_LS((m*(ii-1)+1):m*ii,iii) = ThisBits(end:-1:1);
                    % Bit and symbol detection with MMSE equalizer
                    [junk,ThisQamIdx] = min(abs(Const - X_hat_MMSE(ii,iii)));
                    ThisBits = abs(dec2bin(ThisQamIdx-1,m) - '0');
                    RxBits_MMSE((m*(ii-1)+1):m*ii,iii) = ThisBits(end:-1:1);
                end
            end
            TotalBerNum_ESN = TotalBerNum_ESN + sum(sum(TxBits ~= RxBits_ESN));
            TotalBerNum_LS = TotalBerNum_LS + sum(sum(TxBits ~= RxBits_LS));
            TotalBerNum_MMSE = TotalBerNum_MMSE + sum(sum(TxBits ~= RxBits_MMSE));
            TotalBerNum_Perfect = TotalBerNum_Perfect + sum(sum(TxBits ~= RxBits_Perfect));
            TotalBerDen = TotalBerDen + size(TxBits,1)*size(TxBits,2);
        end    
    end
    % Compute and store the bit error rate(BER) values for the current
    % signal to noise ratio.
    BER_ESN(jj) = TotalBerNum_ESN./TotalBerDen
    BER_LS(jj) = TotalBerNum_LS./TotalBerDen
    BER_MMSE(jj) = TotalBerNum_MMSE./TotalBerDen
    BER_Perfect(jj) = TotalBerNum_Perfect./TotalBerDen;
end
NMSE_ESN_Testing = NMSE_ESN_Testing./NMSE_count;
NMSE_ESN_Training = NMSE_ESN_Training./(NumOfdmSymbols - NMSE_count);
% 
hold off
semilogy(EbNoDB,BER_LS,'o-','LineWidth',1.5);
hold on
semilogy(EbNoDB,BER_ESN,'gd--','LineWidth',1.5);
semilogy(EbNoDB,BER_MMSE,'rs-.','LineWidth',1.5);

grid on
title('64 Neurons in the Reservoir');
xlabel('Signal-to-Noise Ratio[dB]');
ylabel('Bit Error Rate');
legend('LS','ESN','LMMSE');


