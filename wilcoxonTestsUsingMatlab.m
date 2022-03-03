clear variables;

%% Figure 1

load('/home/janaki/Dropbox/GeffenLab/Janaki/uncertainty code/figures/FromProlific/illustrations/PerformanceAccuracyChangesWithFrequencyOfDistractor.mat')

p_Freq = zeros(15);
h_Freq = zeros(15);
zval_Freq = zeros(15);
for i = 2:16
    [p_Freq(i-1),h_Freq(i-1),stats_Freq] = signrank(dict_behaviour.averageBehaviorLow(i,:),dict_behaviour.averageBehaviorHigh(i,:));
    zval_Freq(i-1) = stats_Freq.zval;
end

noContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','NoContextModelFits');

bic_subsampledData_probabilisticModel = table2array(noContextData(:,9));
bic_subsampledData_probabilisticModel = bic_subsampledData_probabilisticModel(~isnan(bic_subsampledData_probabilisticModel));

bic_subsampledData_randomModel = table2array(noContextData(:,20));
bic_subsampledData_randomModel = bic_subsampledData_randomModel(~isnan(bic_subsampledData_randomModel));

bic_subsampledData_signalModel = table2array(noContextData(:,23));
bic_subsampledData_signalModel = bic_subsampledData_signalModel(~isnan(bic_subsampledData_signalModel));

[p_NoContext_RandomVsProbabilistic,h_NoContext_RandomVsProbabilistic,stats_NoContext_RandomVsProbabilistic] = signrank(bic_subsampledData_probabilisticModel+6*log(500), bic_subsampledData_randomModel+2*log(500));
[p_NoContext_SignalVsProbabilistic,h_NoContext_SignalVsProbabilistic,stats_NoContext_SignalVsProbabilistic] = signrank(bic_subsampledData_signalModel+5*log(500), bic_subsampledData_probabilisticModel+6*log(500));

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

%% Figure 5
[~,SheetNames]  = xlsfinfo('clusterResultsForSubsampledData.xls');
nSheets = length(SheetNames);
internalizedBias = readtable('clusterResultsForSubsampledData.xls','Sheet','internalizedBias');

biasLowForSubjectsWithNoAndLowContexts = table2array(internalizedBias(:,5));
biasLowForSubjectsWithNoAndLowContexts = biasLowForSubjectsWithNoAndLowContexts(~isnan(biasLowForSubjectsWithNoAndLowContexts));
biasNoForSubjectsWithNoAndLowContexts = table2array(internalizedBias(:,2));
biasNoForSubjectsWithNoAndLowContexts = biasNoForSubjectsWithNoAndLowContexts(~isnan(biasNoForSubjectsWithNoAndLowContexts));

biasHighForSubjectsWithNoAndHighContexts = table2array(internalizedBias(:,21));
biasHighForSubjectsWithNoAndHighContexts = biasHighForSubjectsWithNoAndHighContexts(~isnan(biasHighForSubjectsWithNoAndHighContexts));
biasNoForSubjectsWithNoAndHighContexts = table2array(internalizedBias(:,2));
biasNoForSubjectsWithNoAndHighContexts = biasNoForSubjectsWithNoAndHighContexts(~isnan(biasHighForSubjectsWithNoAndHighContexts));

[p_biasNoVsLow,h_biasNoVsLow,stats_biasNoVsLow] = signrank((0.5-biasLowForSubjectsWithNoAndLowContexts)*2, (0.5-biasNoForSubjectsWithNoAndLowContexts)*2);
[p_biasNoVsHigh,h_biasNoVsHigh,stats_biasNoVsHigh] = signrank((biasHighForSubjectsWithNoAndHighContexts-0.5)*2, (0.5-biasNoForSubjectsWithNoAndHighContexts)*2);

%% Figure 7
lowContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','LowContextModelFits');

bic_lowContext_probabilisticModel_all = table2array(lowContextData(:,9));
bic_lowContext_probabilisticModel_all = bic_lowContext_probabilisticModel_all(~isnan(bic_lowContext_probabilisticModel_all));
bic_lowContext_probabilisticModel = bic_lowContext_probabilisticModel_all(1:2:end);

bic_lowContext_randomModel = table2array(lowContextData(:,20));
bic_lowContext_randomModel = bic_lowContext_randomModel(~isnan(bic_lowContext_randomModel));

bic_lowContext_signalModel = table2array(lowContextData(:,23));
bic_lowContext_signalModel = bic_lowContext_signalModel(~isnan(bic_lowContext_signalModel));

size_lowContext = table2array(lowContextData(:,26));
size_lowContext = size_lowContext(~isnan(size_lowContext));

bic_lowContext_probabilisticModel([26,46]) = [];
bic_lowContext_randomModel([26,46]) = [];
bic_lowContext_signalModel([26,46]) = [];
size_lowContext([26,46]) = [];

highContextData = readtable('clusterResultsForSubsampledData.xls','Sheet','HighContextModelFits');

bic_highContext_probabilisticModel = table2array(highContextData(:,9));
bic_highContext_probabilisticModel = bic_highContext_probabilisticModel(~isnan(bic_highContext_probabilisticModel));

bic_highContext_randomModel = table2array(highContextData(:,20));
bic_highContext_randomModel = bic_highContext_randomModel(~isnan(bic_highContext_randomModel));

bic_highContext_signalModel = table2array(highContextData(:,23));
bic_highContext_signalModel = bic_highContext_signalModel(~isnan(bic_highContext_signalModel));

size_highContext = table2array(highContextData(:,26));
size_highContext = size_highContext(~isnan(size_highContext));

bic_highContext_probabilisticModel([21,41]) = [];
bic_highContext_randomModel([21,41]) = [];
bic_highContext_signalModel([21,41]) = [];
size_highContext([21,41]) = [];

[p_LowContext_RandomVsProbabilistic,h_LowContext_RandomVsProbabilistic,stats_LowContext_RandomVsProbabilistic] = signrank(bic_lowContext_probabilisticModel+6*log(size_lowContext), bic_lowContext_randomModel+2*log(size_lowContext));
[p_LowContext_SignalVsProbabilistic,h_LowContext_SignalVsProbabilistic,stats_LowContext_SignalVsProbabilistic] = signrank(bic_lowContext_signalModel+5*log(size_lowContext), bic_lowContext_probabilisticModel+6*log(size_lowContext));

[p_HighContext_RandomVsProbabilistic,h_HighContext_RandomVsProbabilistic,stats_HighContext_RandomVsProbabilistic] = signrank(bic_highContext_probabilisticModel+6*log(size_highContext), bic_highContext_randomModel+2*log(size_highContext));
[p_HighContext_SignalVsProbabilistic,h_HighContext_SignalVsProbabilistic,stats_HighContext_SignalVsProbabilistic] = signrank(bic_highContext_signalModel+5*log(size_highContext), bic_highContext_probabilisticModel+6*log(size_highContext));



