clear variables;

%% Figures 2,4

load('/home/janaki/Dropbox/GeffenLab/Janaki/uncertaintyCode/figures/FromProlific/illustrations/PerformanceAccuracyChangesWithFrequencyOfDistractor.mat')

p_Freq = zeros(15);
h_Freq = zeros(15);
zval_Freq = zeros(15);
for i = 2:16
    [p_Freq(i-1),h_Freq(i-1),stats_Freq] = signrank(dict_behaviour.averageBehaviorLow(i,:),dict_behaviour.averageBehaviorHigh(i,:));
    zval_Freq(i-1) = stats_Freq.zval;
end

noContextDataVer = readtable('clusterResultsForSubsampledData.xls','Sheet','NoContextModelFits_veridicalPar');

bic_subsampledData_probabilisticModelVer = table2array(noContextDataVer(:,13));
bic_subsampledData_probabilisticModelVer = bic_subsampledData_probabilisticModelVer(~isnan(bic_subsampledData_probabilisticModelVer));

bic_subsampledData_randomModel = table2array(noContextDataVer(:,25));
bic_subsampledData_randomModel = bic_subsampledData_randomModel(~isnan(bic_subsampledData_randomModel));

bic_subsampledData_signalModelVer = table2array(noContextDataVer(:,22));
bic_subsampledData_signalModelVer = bic_subsampledData_signalModelVer(~isnan(bic_subsampledData_signalModelVer));

[p_NoContext_RandomVsProbabilisticVer,h_NoContext_RandomVsProbabilisticVer,stats_NoContext_RandomVsProbabilisticVer] = signrank(2*bic_subsampledData_probabilisticModelVer+3*log(600), 2*bic_subsampledData_randomModel+2*log(600));
[p_NoContext_SignalVsProbabilisticVer,h_NoContext_SignalVsProbabilisticVer,stats_NoContext_SignalVsProbabilisticVer] = signrank(2*bic_subsampledData_signalModelVer+2*log(600), 2*bic_subsampledData_probabilisticModelVer+3*log(600));

noContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','NoContextModelFits');

bic_subsampledData_probabilisticModel = table2array(noContextData(:,9));
bic_subsampledData_probabilisticModel = bic_subsampledData_probabilisticModel(~isnan(bic_subsampledData_probabilisticModel));

bic_subsampledData_signalModel = table2array(noContextData(:,28));
bic_subsampledData_signalModel = bic_subsampledData_signalModel(~isnan(bic_subsampledData_signalModel));

[p_NoContext_ProbabilisticComp,h_NoContext_ProbabilisticComp,stats_NoContext_ProbabilisticComp] = signrank(2*bic_subsampledData_probabilisticModelVer+3*log(600), 2*bic_subsampledData_probabilisticModel+6*log(600));
[p_NoContext_SignalComp,h_NoContext_SignalComp, stats_NoContext_SignalComp] = signrank(2*bic_subsampledData_signalModelVer+2*log(600), 2*bic_subsampledData_signalModel+5*log(600));

meanLowValues_probabilisticModel = table2array(noContextData(:,2));
meanLowValues_probabilisticModel = meanLowValues_probabilisticModel(~isnan(meanLowValues_probabilisticModel));

meanHighValues_probabilisticModel = table2array(noContextData(:,3));
meanHighValues_probabilisticModel = meanHighValues_probabilisticModel(~isnan(meanHighValues_probabilisticModel));

ssValues_probabilisticModel = table2array(noContextData(:,5));
ssValues_probabilisticModel = ssValues_probabilisticModel(~isnan(ssValues_probabilisticModel));

pBackValues_probabilisticModel = table2array(noContextData(:,6));
pBackValues_probabilisticModel = pBackValues_probabilisticModel(~isnan(pBackValues_probabilisticModel));

pLowValues_probabilisticModel = table2array(noContextData(:,7));
pLowValues_probabilisticModel = pLowValues_probabilisticModel(~isnan(pLowValues_probabilisticModel));

disp(["Mean and std of means of low gaussian",mean(meanLowValues_probabilisticModel), std(meanLowValues_probabilisticModel)])
disp(["Mean and std of means of high gaussian",mean(meanHighValues_probabilisticModel), std(meanHighValues_probabilisticModel)])
disp(["Mean and std of sensory sigmas",mean(ssValues_probabilisticModel), std(ssValues_probabilisticModel)])
disp(["Mean and std of probability of background",mean(pBackValues_probabilisticModel), std(pBackValues_probabilisticModel)])
disp(["Mean and std of probability of low",mean(pLowValues_probabilisticModel), std(pLowValues_probabilisticModel)])

distractorPerformanceData = readtable('clusterResultsForSubsampledData.xls','Sheet','PerformanceVsNumDistractors');

OneIrrelevantToneAccuracy = table2array(distractorPerformanceData(1:56,11));
TwoIrrelevantTonesAccuracy = table2array(distractorPerformanceData(1:56,12));
figure(); hold on;
plot(OneIrrelevantToneAccuracy, TwoIrrelevantTonesAccuracy,'.')
plot(50:90,50:90,'k--')

OneIrrelevantToneAccuracy([30,35]) = [];
TwoIrrelevantTonesAccuracy([30,35]) = [];
[p_OneToneVsTwoTonesAccuracy, h_OneToneVsTwoTonesAccuracy, stats_OneToneVsTwoTonesAccuracy] = signrank(OneIrrelevantToneAccuracy, TwoIrrelevantTonesAccuracy);

%% extended data Fig. 5
lowContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','BiasedLowModelFits_veridicalPar');

bic_lowContext_probabilisticModel = table2array(lowContextData(:,14));
bic_lowContext_probabilisticModel = bic_lowContext_probabilisticModel(~isnan(bic_lowContext_probabilisticModel));

bic_lowContext_randomModel = table2array(lowContextData(:,19));
bic_lowContext_randomModel = bic_lowContext_randomModel(~isnan(bic_lowContext_randomModel));

bic_lowContext_signalModel = table2array(lowContextData(:,22));
bic_lowContext_signalModel = bic_lowContext_signalModel(~isnan(bic_lowContext_signalModel));

size_lowContext = table2array(lowContextData(:,25));
size_lowContext = size_lowContext(~isnan(size_lowContext));

highContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','BiasedHighModelFits_veridicalPa');

bic_highContext_probabilisticModel = table2array(highContextData(:,14));
bic_highContext_probabilisticModel = bic_highContext_probabilisticModel(~isnan(bic_highContext_probabilisticModel));

bic_highContext_randomModel = table2array(highContextData(:,19));
bic_highContext_randomModel = bic_highContext_randomModel(~isnan(bic_highContext_randomModel));

bic_highContext_signalModel = table2array(highContextData(:,22));
bic_highContext_signalModel = bic_highContext_signalModel(~isnan(bic_highContext_signalModel));

size_highContext = table2array(highContextData(:,25));
size_highContext = size_highContext(~isnan(size_highContext));

[p_LowContext_RandomVsProbabilistic,h_LowContext_RandomVsProbabilistic,stats_LowContext_RandomVsProbabilistic] = signrank(2*bic_lowContext_probabilisticModel+3*log(size_lowContext), 2*bic_lowContext_randomModel+2*log(size_lowContext));
[p_HighContext_RandomVsProbabilistic,h_HighContext_RandomVsProbabilistic,stats_HighContext_RandomVsProbabilistic] = signrank(2*bic_highContext_probabilisticModel+3*log(size_highContext), 2*bic_highContext_randomModel+2*log(size_highContext));

bic_lowContext_probabilisticModel([21,41]) = [];
bic_lowContext_randomModel([21,41]) = [];
bic_lowContext_signalModel([21,41]) = [];
size_lowContext([21,41]) = [];

bic_highContext_probabilisticModel([21,41]) = [];
bic_highContext_randomModel([21,41]) = [];
bic_highContext_signalModel([21,41]) = [];
size_highContext([21,41]) = [];

[p_LowContext_SignalVsProbabilistic,h_LowContext_SignalVsProbabilistic,stats_LowContext_SignalVsProbabilistic] = signrank(2*bic_lowContext_signalModel+2*log(size_lowContext), 2*bic_lowContext_probabilisticModel+3*log(size_lowContext));
[p_HighContext_SignalVsProbabilistic,h_HighContext_SignalVsProbabilistic,stats_HighContext_SignalVsProbabilistic] = signrank(2*bic_highContext_signalModel+2*log(size_highContext), 2*bic_highContext_probabilisticModel+3*log(size_highContext));



