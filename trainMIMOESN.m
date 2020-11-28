function [trainedEsn, Delay, Delay_Minn,  Delay_Maxx, nForgetPoints, NMSE_ESN] = trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, ...
    CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP)
% This function trains the ESN for a 2x2 MIMO system for various values of
% the output delays at the 4 output nodes denoted by d_1, d_2, d_3, d_4.
% For each delay quadruple ut computes the NMSE. Then it picks the delay
% quadruple with the lowest NMSE, and trains the esn with the chosen
% delays. First a delay look-up-table with different delays is generates.
% If the flag is set we consider a larger number of delays quadruples. If
% it is not set, we only consider delays of the form d_1 = d_2 = d_3 = d_4.
% This approach reduces the training time greatly.
if(DelayFlag)
    Delay_LUT = zeros((Max_Delay+1-Min_Delay)^2,4);
    count = 0;
    temp = zeros(size(Delay_LUT,1),1);
    for ii = Min_Delay:Max_Delay
        for jj = Min_Delay:Max_Delay
            for kk = Min_Delay:Max_Delay
                for ll = Min_Delay:Max_Delay
                      count = count + 1;
                      Delay_LUT(count, :) = [ii jj kk ll];
                     if(abs(ii-jj) > 2)
                         temp(count) = 1;
                     elseif (abs(kk-ll) > 2)
                          temp(count) = 1;
                     elseif (abs(ii-kk) > 2)
                         temp(count) = 1;
                     elseif (abs(ii-ll) > 2)
                         temp(count) = 1;
                     elseif (abs(jj-kk) > 2)
                         temp(count) = 1;
                     elseif (abs(jj-ll) > 2)
                         temp(count) = 1;
                     end
                end
            end
        end
    end
    Delay_LUT(temp>0,:) = [];
else
     Delay_LUT = zeros((Max_Delay+1-Min_Delay),4);   
     for jjjj = 1:Max_Delay+1
        Delay_LUT(jjjj,:) = (jjjj-1).*ones(1,4);
     end
end
% Compute the max and teh min of ach dealy row. 
[Delay_Max Max_Idx] = max(Delay_LUT');
Delay_Max = Delay_Max(:);
[Delay_Min Min_Idx] = min(Delay_LUT');
Delay_Min = Delay_Min(:);

% Train the esn with different delays and store the training NMSE.
NMSE_ESN_Training = zeros(size(Delay_LUT,1),1);
for jjj = 1:size(Delay_LUT,1)    
    Curr_Delay = Delay_LUT(jjj, :);
    
    ESN_input = zeros(N + Delay_Max(jjj) + CyclicPrefixLen,N_t*2);
    ESN_output = zeros(N + Delay_Max(jjj)+ CyclicPrefixLen,N_t*2);
    % The ESN input
    ESN_input(:,1) = [real(y_CP(:,1)); zeros(Delay_Max(jjj),1)];
    ESN_input(:,2) = [imag(y_CP(:,1)); zeros(Delay_Max(jjj),1)];
    ESN_input(:,3) = [real(y_CP(:,2)); zeros(Delay_Max(jjj),1)];
    ESN_input(:,4) = [imag(y_CP(:,2)); zeros(Delay_Max(jjj),1)];
    % The ESN output
    ESN_output((Curr_Delay(1)+1):(Curr_Delay(1) + N + CyclicPrefixLen),1) = real(x_CP(:,1));
    ESN_output((Curr_Delay(2)+1):(Curr_Delay(2) + N + CyclicPrefixLen),2) = imag(x_CP(:,1));
    ESN_output((Curr_Delay(3)+1):(Curr_Delay(3) + N + CyclicPrefixLen),3) = real(x_CP(:,2));
    ESN_output((Curr_Delay(4)+1):(Curr_Delay(4) + N + CyclicPrefixLen),4) = imag(x_CP(:,2));
    % Train the ESN
    nForgetPoints = Delay_Min(jjj) + CyclicPrefixLen;
    [trainedEsn, stateMatrix] = train_esn(ESN_input, ESN_output, esn, nForgetPoints) ;
    % Get the ESN output corresponding to the training input
    x_hat_ESN_temp = test_esn(ESN_input, trainedEsn, nForgetPoints);
    % Put the real and the imaginary parts of each transmitted stream together
    x_hat_ESN(:,1) = x_hat_ESN_temp(Curr_Delay(1) - Delay_Min(jjj)+1: Curr_Delay(1) - Delay_Min(jjj)+N, 1) ...
        + j.*x_hat_ESN_temp(Curr_Delay(2) - Delay_Min(jjj)+1: Curr_Delay(2) - Delay_Min(jjj)+N, 2);
    x_hat_ESN(:,2) = x_hat_ESN_temp(Curr_Delay(3) - Delay_Min(jjj)+1: Curr_Delay(3) - Delay_Min(jjj)+N, 3) ...
        + j.*x_hat_ESN_temp(Curr_Delay(4) - Delay_Min(jjj)+1: Curr_Delay(4) - Delay_Min(jjj)+N, 4);
    % Compute the training NMSE
    x = x_CP(IsiDuration:end,:);
    NMSE_ESN_Training(jjj) = ((x_hat_ESN(:,1) - x(:,1) )'*(x_hat_ESN(:,1)  - x(:,1) ))./(x(:,1)'*x(:,1))...
        + ((x_hat_ESN(:,2) - x(:,2) )'*(x_hat_ESN(:,2)  - x(:,2) ))./(x(:,2)'*x(:,2)); 
end
% Find the delay row that minimizes the NMSE.
[NMSE_ESN Delay_Idx] = min(NMSE_ESN_Training);
Delay = Delay_LUT(Delay_Idx,:);
% Train the esn with the optilam delay quadruple. 
ESN_input = zeros(N + Delay_Max(Delay_Idx) + CyclicPrefixLen,N_t*2);
ESN_output = zeros(N + Delay_Max(Delay_Idx)+ CyclicPrefixLen,N_t*2);

ESN_input(:,1) = [real(y_CP(:,1)); zeros(Delay_Max(Delay_Idx),1)];
ESN_input(:,2) = [imag(y_CP(:,1)); zeros(Delay_Max(Delay_Idx),1)];
ESN_input(:,3) = [real(y_CP(:,2)); zeros(Delay_Max(Delay_Idx),1)];
ESN_input(:,4) = [imag(y_CP(:,2)); zeros(Delay_Max(Delay_Idx),1)];

ESN_output((Delay(1)+1):(Delay(1) + N + CyclicPrefixLen),1) = real(x_CP(:,1));
ESN_output((Delay(2)+1):(Delay(2) + N + CyclicPrefixLen),2) = imag(x_CP(:,1));
ESN_output((Delay(3)+1):(Delay(3) + N + CyclicPrefixLen),3) = real(x_CP(:,2));
ESN_output((Delay(4)+1):(Delay(4) + N + CyclicPrefixLen),4) = imag(x_CP(:,2));
nForgetPoints = Delay_Min(Delay_Idx) + CyclicPrefixLen;
[trainedEsn, stateMatrix] = train_esn(ESN_input, ESN_output, esn, nForgetPoints) ;
Delay_Minn = Delay_Min(Delay_Idx);
Delay_Maxx = Delay_Max(Delay_Idx);