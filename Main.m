% All Rights Reserved.
% Copyright (c) 2024 Roneel V. Sharan.
% University of Essex, United Kingdom.
% Email: roneel.sharan@essex.ac.uk.
%
% Citation for this work:
% [1] R. V. Sharan, H. Takeuchi, A. Kishi and Y. Yamamoto, "Macro-sleep 
% staging with ECG-derived instantaneous heart rate and respiration signals 
% and multi-input 1-D CNN-BiGRU," IEEE Transactions on Instrumentation and 
% Measurement, vol. 73, pp. 1-12, 2024, Art no. 2535212. 
% DOI: https://doi.org/10.1109/TIM.2024.3481551.
%
% The codes below require the PhysioNet Cardiovascular Signal Toolbox and
% The WFDB Software Package. These are available from PhysioNet. The
% example below uses the MIT-BIH Polysomnographic Database which is also
% available from PhysioNet.

clc

%% Number of Sleep Stages
ClassificationProblem = '3Class'; % '2Class' '3Class' '4Class' '5Class'

%% Read ECG Signal
SubID = 'slp04';
DataPath = '.\mit-bih-polysomnographic-database-1.0.0\';
[Signal, Fs] = rdsamp([DataPath, SubID]);
[SigInfo] = wfdbdesc([DataPath, SubID]);
ECGSignal = Signal(:, strcmp({SigInfo.Description}, 'ECG'));
[ST.Ann, ST.AnnType, ST.SubType, ST.Chan, ST.Num, ST.Comments] = rdann([DataPath, SubID], 'st');
ST.Comments = cellfun(@(x) x(1), ST.Comments, 'UniformOutput', false);

%% IHR
HRVparams.Fs = Fs;
HRVparams.PeakDetect.REF_PERIOD = 0.250;
HRVparams.PeakDetect.THRES = .6;
HRVparams.PeakDetect.fid_vec = [];
HRVparams.PeakDetect.SIGN_FORCE = [];
HRVparams.PeakDetect.debug = 0;
HRVparams.PeakDetect.ecgType = 'MECG';
HRVparams.PeakDetect.windows = 15;
HRVparams.gen_figs = 0;
HRVparams.preprocess.gaplimit = 2;
HRVparams.preprocess.per_limit = 0.2; 
HRVparams.preprocess.method_outliers = 'rem';
HRVparams.preprocess.lowerphysiolim = 60/160;
HRVparams.preprocess.upperphysiolim = 60/30;
HRVparams.preprocess.method_unphysio = 'rem';

ECG.Ann = run_qrsdet_by_seg(ECGSignal, HRVparams);
ECG.Ann = ECG.Ann';
IHRLoc = ECG.Ann./Fs;
IHRLoc = IHRLoc(2:end);
IBI = diff(ECG.Ann)./Fs;
[IBI, IHRLoc] = RRIntervalPreprocess(IBI, IHRLoc, [], HRVparams);
IHR = 1./IBI;
IHR = zscore(IHR);
ResamplePts = 1/2:1/2:length(ECGSignal)/Fs;
ECG.AnnResample = interp1(IHRLoc, IHR, ResamplePts, 'linear');
ECG.AnnResample = fillmissing(ECG.AnnResample, 'linear');

%% EDR
ECGSignal = zscore(ECGSignal);
y = edr(0, ECGSignal, IHRLoc, Fs);
EDR.Ann = y(:,1);
EDR.AnnType = y(:,2);
EDR.AnnType = zscore(EDR.AnnType);
EDR.AnnTypeResample = interp1(EDR.Ann, EDR.AnnType, ResamplePts, 'linear');
EDR.AnnTypeResample = fillmissing(EDR.AnnTypeResample, 'linear');

%% Epoch
EpochDuration = 30;
nEpoch = min([numel(ST.Ann), floor(length(ECGSignal)/(EpochDuration*Fs))]);
EpochPoints = [1:nEpoch]';
for e = 1:nEpoch
    IdxEpoch = sort(knnsearch(EpochPoints, e, 'K', 5), 'ascend');
    NearestEpochs = EpochPoints(IdxEpoch);
    eStart = min(NearestEpochs);
    eStop = max(NearestEpochs);

    %% IHR and EDR Signal for Deep Learning
    windowTimeStart = (ST.Ann(eStart) - 1)/Fs;
    windowTimeStop = (ST.Ann(eStop) + (EpochDuration*Fs) - 1)/Fs;
    
    ECGDerivedSignals(1,:,1,e) = ECG.AnnResample(ResamplePts > windowTimeStart & ResamplePts <= windowTimeStop);
    ECGDerivedSignals(1,:,2,e) = EDR.AnnTypeResample(ResamplePts > windowTimeStart & ResamplePts <= windowTimeStop);
end

%% Data Use
idxWakeSleep = ismember(ST.Comments, {'W', '1', '2', '3', '4', 'R'});
ClassUse = ST.Comments(idxWakeSleep,:);
switch ClassificationProblem
    case '2Class'
        ClassNum(ismember(ClassUse, {'W'}), 1) = 0;
        ClassNum(ismember(ClassUse, {'1', '2', '3', '4', 'R'}), 1) = 1;
        Classes = [0, 1];
    case '3Class'
        ClassNum(ismember(ClassUse, {'W'}), 1) = 0;
        ClassNum(ismember(ClassUse, {'1', '2', '3', '4'}), 1) = 1;
        ClassNum(ismember(ClassUse, {'R'}), 1) = 2;
        Classes = [0, 1, 2];
    case '4Class'
        ClassNum(ismember(ClassUse, {'W'}), 1) = 0;
        ClassNum(ismember(ClassUse, {'1', '2'}), 1) = 1;
        ClassNum(ismember(ClassUse, {'3', '4'}), 1) = 2;
        ClassNum(ismember(ClassUse, {'R'}), 1) = 3;
        Classes = [0, 1, 2, 3];
    case '5Class'
        ClassNum(ismember(ClassUse, {'W'}), 1) = 0;
        ClassNum(ismember(ClassUse, {'1'}), 1) = 1;
        ClassNum(ismember(ClassUse, {'2'}), 1) = 2;
        ClassNum(ismember(ClassUse, {'3', '4'}), 1) = 3;
        ClassNum(ismember(ClassUse, {'R'}), 1) = 4;
        Classes = [0, 1, 2, 3, 4];
end
ActualClass = ClassNum;
ECGDerivedSignals = ECGDerivedSignals(:,:,:,idxWakeSleep);

%% Load Pretrained Nets (Trained on MGH dataset)
IHRModel = load([cd, '/TrainedNets/IHR', ClassificationProblem, '.mat']);
EDRModel = load([cd, '/TrainedNets/EDR', ClassificationProblem, '.mat']);
SMXModel = load([cd, '/TrainedNets/SMX', ClassificationProblem, '.mat']);

%% Network Activation
IHRActivation = activations(IHRModel.TrainModel, ECGDerivedSignals(:,:,1,:), 'Dense', 'MiniBatchSize', 32, 'OutputAs', 'channels');
EDRActivation = activations(EDRModel.TrainModel, ECGDerivedSignals(:,:,2,:), 'Dense', 'MiniBatchSize', 32, 'OutputAs', 'channels');
CombinedActivations = [IHRActivation{1}; EDRActivation{1}]';

%% Classify
[~, Score] = classify(SMXModel.TrainModel, CombinedActivations);
[~, ClassIdx] = max(Score, [], 2);
PredictedClass = Classes(ClassIdx)';
CM = confusionmat(ActualClass, PredictedClass)
ClassAccuracy = diag(CM)./sum(CM, 2)