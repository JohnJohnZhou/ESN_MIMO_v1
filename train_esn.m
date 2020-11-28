 function [trained_esn, stateCollection, in] = ...
    train_esn(trainInput, trainOutput , esn, nForgetPoints)
% TRAIN_ESN Trains the output weights of an ESN 
% In the offline case, it computes the weights using the method
% esn.methodWeightCompute(for ex linear regression using pseudo-inverse)
% In the online case, RLS is being used. 
% 
% inputs:
% trainInput = input vector of size nTrainingPoints x nInputDimension
% trainOutput = teacher vector of size nTrainingPoints x
% nOutputDimension
% esn = an ESN structure, through which we run our input sequence
% nForgetPoints - the first nForgetPoints will be disregarded

%
% outputs: 
% trained_esn = an Esn structure with the option trained = 1 and 
% outputWeights set. 
% stateCollection = matrix of size (nTrainingPoints-nForgetPoints) x
% nInputUnits + nInternalUnits 
% stateCollectMat(i,j) = internal activation of unit j after the 
% (i + nForgetPoints)th training point has been presented to the network
% teacherCollection is a nSamplePoints * nOuputUnits matrix that keeps
% the expected output of the ESN
% teacherCollection is the transformed(scaled, shifted etc) output see
% compute_teacher for more documentation

%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision 1, June 30, 2006, H. Jaeger
% Revision 2, Feb 23, 2007, H. Jaeger


trained_esn = esn;
switch trained_esn.learningMode
    case 'offline_singleTimeSeries'
        % trainInput and trainOutput each represent a single time series in
        % an array of size sequenceLength x sequenceDimension
        if strcmp(trained_esn.type, 'twi_esn')
            if size(trainInput,2) > 1
                trained_esn.avDist = ...
                    mean(sqrt(sum(((trainInput(2:end,:) - trainInput(1:end - 1,:))').^2)));
            else
                trained_esn.avDist = mean(abs(trainInput(2:end,:) - trainInput(1:end - 1,:)));
            end
        end
        [stateCollection in]= compute_statematrix(trainInput, trainOutput, trained_esn, nForgetPoints) ; 
        teacherCollection = compute_teacher(trainOutput, trained_esn, nForgetPoints) ;             
        trained_esn.outputWeights = feval(trained_esn.methodWeightCompute, stateCollection, teacherCollection) ;
        
    case  'offline_multipleTimeSeries'   
        % trainInput and trainOutput each represent a collection of K time
        % series, given in cell arrays of size K x 1, where each cell is an
        % array of size individualSequenceLength x sequenceDimension
        
        % compute total size of sample points to be used
        sampleSize = 0;
        nTimeSeries = size(trainInput, 1);
        for i = 1:nTimeSeries
            sampleSize = sampleSize + size(trainInput{i,1},1) - max([0, nForgetPoints]);
        end
        
        % collect input+reservoir states into stateCollection
        stateCollection = zeros(sampleSize, trained_esn.nInputUnits + trained_esn.nInternalUnits);
        collectIndex = 1;
        for i = 1:nTimeSeries
            if strcmp(trained_esn.type, 'twi_esn')
                if size(trainInput{i,1},2) > 1
                    trained_esn.avDist = ...
                        mean(sqrt(sum(((trainInput{i,1}(2:end,:) - trainInput{i,1}(1:end - 1,:))').^2)));
                else
                    trained_esn.avDist = mean(abs(trainInput{i,1}(2:end,:) - trainInput{i,1}(1:end - 1,:)));
                end
            end           
            stateCollection_i = ...
                compute_statematrix(trainInput{i,1}, trainOutput{i,1}, trained_esn, nForgetPoints);
            l = size(stateCollection_i, 1);
            stateCollection(collectIndex:collectIndex+l-1, :) = stateCollection_i;
            collectIndex = collectIndex + l;
        end
        
        % collect teacher signals (including applying the inverse output
        % activation function) into teacherCollection
        teacherCollection = zeros(sampleSize, trained_esn.nOutputUnits);
        collectIndex = 1;
        for i = 1:nTimeSeries
            teacherCollection_i = ...
                compute_teacher(trainOutput{i,1}, trained_esn, nForgetPoints);
            l = size(teacherCollection_i, 1);
            teacherCollection(collectIndex:collectIndex+l-1, :) = teacherCollection_i;
            collectIndex = collectIndex + l;
        end
        
        % compute output weights
        trained_esn.outputWeights = ...
            feval(trained_esn.methodWeightCompute, stateCollection, teacherCollection) ;
        
        
    case 'online'
        nSampleInput = length(trainInput);  
        stateCollection = zeros(nSampleInput, trained_esn.nInternalUnits + trained_esn.nInputUnits);
        SInverse = 1 / trained_esn.RLS_delta * eye(trained_esn.nInternalUnits + trained_esn.nInputUnits) ; 
        totalstate = zeros(trained_esn.nTotalUnits,1);
        internalState = zeros(trained_esn.nInternalUnits,1) ; 
        error = zeros(nSampleInput , 1) ; 
        weights = zeros(nSampleInput , 1) ; 
        for iInput = 1 : nSampleInput
            if trained_esn.nInputUnits > 0
                in = [diag(trained_esn.inputScaling) * trainInput(iInput,:)' + esn.inputShift];  % in is column vector
            else in = [];
            end
            
            %write input into totalstate
            if esn.nInputUnits > 0
                totalstate(esn.nInternalUnits+1:esn.nInternalUnits+esn.nInputUnits) = in;
            end
            
            % update totalstate except at input positions
            
            % the internal state is computed based on the type of the network
            switch esn.type
                case 'plain_esn'
                    typeSpecificArg = [];
                case 'leaky_esn'
                    typeSpecificArg = [];
                case 'twi_esn'
                    if  esn.nInputUnits == 0
                        error('twi_esn cannot be used without input to ESN');
                    end
                    typeSpecificArg = esn.avDist;                
            end
            internalState = feval(trained_esn.type , totalstate, trained_esn, typeSpecificArg ) ;                             
            netOut = feval(trained_esn.outputActivationFunction,trained_esn.outputWeights*[internalState;in]);       
            totalstate = [internalState;in;netOut];            
            state = [internalState;in] ; 
            stateCollection(iInput, :) = state';
            phi = state' * SInverse ;
            %            u = SInverse * state ; 
            %            k = 1 / (lambda + state'*u)*u ; 
            k = phi'/(trained_esn.RLS_lambda + phi * state );
            e = trained_esn.teacherScaling * trainOutput(iInput,1) + trained_esn.teacherShift - netOut(1) ; 
            % collect the error that will be plotted
            error(iInput , 1 ) = e*e ; 
            % update the weights 
            trained_esn.outputWeights(1,:) = trained_esn.outputWeights(1,:) + (k*e)' ;             
            % collect the weights for plotting 
            weights(iInput , 1) = sum(abs(trained_esn.outputWeights(1,:))) ; 
            %            SInverse = 1 / lambda * (SInverse - k*(state' * SInverse)) ;                         
            SInverse = ( SInverse - k * phi ) / trained_esn.RLS_lambda ;
        end       
        
end

trained_esn.trained = 1 ;          


