%% run BADS with zero pDist


%parameter ranges for initializing p0
mu1_range = 2:0.1:3;
mu2_range = 2:0.1:3;
sigma_range = 0.1:0.1:0.6;
prior_range = 0:0.1:1;

options = bads('defaults');
options.Display = 'final';

%subject files and sensory sigmas
subject_files = dir('*.csv');
participants = [0.15 0.19 0.09 0.15 0.14 0.26 0.37 0.15 0.43 0.28];

%iterate over participants
for n = 1:10
    fprintf('Participant: %i\n', n);

    [exp_tones,~ , test_ans, ~, ...
        ~, ~] = extractTonesChoice(subject_files(n).name);

    sigma_sens = participants(n);
    
    lb = [1, 1, 0.05, sigma_sens, 1, 0];
    ub = [5, 5, 0.6, sigma_sens, 1, 1];
    plb = [2.66, 2.36, 0.1, sigma_sens, 1, 0.3];
    pub = [3.06, 2.75, 0.5, sigma_sens, 1, 0.8];
    num_iters = 50;

    %iterate over different p0 
    for i = 1:num_iters

        fprintf('Iteration: %i\n', i);

        ind_1 = randi([1 length(mu1_range)]);
        ind_2 = randi([1 length(mu2_range)]);
        ind_3 = randi([1 length(sigma_range)]);
        ind_4 = randi([1 length(prior_range)]);

        p0 = [mu1_range(ind_1) mu2_range(ind_2) sigma_range(ind_3) sigma_sens 1 prior_range(ind_4)];

        [P,ll(i)] = bads(@(p) getLL(p,exp_tones, test_ans),p0,lb,ub,plb,pub,options);
        P_subject{i} = P;
        phv{i} = generativeModel(P(1), P(2), P(3), P(4), P(5), P(6), exp_tones,1);
    end
    
    %store loglikelihood, fitted parameters, psychometric curves
    loglike{n} = ll;
    P_all{n} = P_subject;
    psychocurve{n} = phv;

end
