%===============================================================================
% SLOTS Provides task functions for the n-armed bandit task from
%   Daw et al (2006). Cortical substrates for exploratory decisions in humans.
%       Nature, 441(7095), 876–879.
%
%   - Here, we have generalized the task to include N-arms rather than 4.
%
% (c)2016 Abraham Nunes; Dalhousie University, Halifax NS, Canada
%===============================================================================

classdef slots
    methods (Static)

        function results = vanilla(subjects, taskparams)
            nactions = taskparams.nactions;

            rdecay    = 0.9836; %decay parameter
            rdecayctr = 0.50;     %decay center
            dsd       = 0.028;    %diffusion noise sd
            rsd       = 0.04;      %sd of gaussian for reward draws

            results = struct();

            for i = 1:subjects.N
                Q = zeros(1, nactions);
                alpha = subjects.params(i, 1);
                beta  = subjects.params(i, 2);

                rpaths = unifrnd(0, 1, [1, 4]);%reward payoff paths
                for t = 1:taskparams.ntrials
                    s = 1;
                    a = softmaxaction(beta.*Q);
                    r = normrnd(rpaths(a), rsd, 1);

                    rpe = r - Q(1, a);
                    %Q = (1-alpha)*Q;
                    Q(1, a) = Q(1, a) + alpha*rpe;

                    rpaths = rdecay*rpaths + (1-rdecay)*rdecayctr + normrnd(0, dsd, [1, nactions]);

                    %record trial in results
                    results(i).S(t, 1)      = s;
                    results(i).A(t, 1)      = a;
                    results(i).R(t, 1)      = r;
                    results(i).RPE(t, 1)    = rpe;
                    results(i).rpaths(t, :) = rpaths;
                    results(i).Q(t, :)      = Q;
                end
            end
            disp(['----- FINISHED SIMULATING ', num2str(subjects.N), ' SUBJECTS -----']);
        end

    end
end
