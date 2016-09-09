%===============================================================================
% Generate simulated data from a Two-Step taskparams
%
% INPUTS:
%       taskparams  = structure containing taskparams specs with the following fields:
%           .ntrials  = number of trials (integer)
%           .nstates  = number of states (integer)
%           .nactions = number of actions (integer)
%           .pReward  = reward probability matrix [nstates by nAction]
%           .rewards  = reward matrix [nstates by nAction]
%           .pTrans   = transition probability matrix
%           .taskparams     = taskparams function handle
%       subjects = structure with subject specs
%           .N          = number of subjects (integer)
%           .K          = number of parameters (integer)
%           .paramNames = {1 by K} cell array of parameter name strings
%           .params     = [N by K] array of subject parameters (K params)
%           .Learn      = learning model function
%           .Act        = observation model function (action selection)
%
%
% OUTPUTS:
%   results = [1 by N] struct with the following fields (for i'th subject):
%     results(i).S = [1 by T] vector of states observed by the subject
%     results(i).A = [1 by T] vector of actions selected by the subject
%     results(i).R = [1 by T] vector of rewards received by the subject
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada
%===============================================================================

classdef twostep
    methods (Static)

    function results = vanilla(subjects, taskparams)

        % Gaussian process path parameters
        dt        = sqrt(1/taskparams.ntrials);

        % Initialize output structure
        results = struct();

        for i = 1:subjects.N
            alpha        = subjects.params(i, 1);
            beta         = subjects.params(i, 2);
            lambda       = subjects.params(i, 3);
            rho          = subjects.params(i, 4);
            omega        = subjects.params(i, 5);
            Q            = zeros(taskparams.nstates, taskparams.nactions);
            Qmb          = zeros(taskparams.nstates, taskparams.nactions);
            Qf           = zeros(taskparams.nstates, taskparams.nactions);
            T            = zeros(taskparams.nstates, taskparams.nactions, taskparams.nstates);
            paths        = unifrnd(0.25, 0.75, [1, 4]);

            for t = 1:taskparams.ntrials
                results(i).paths(t,:) = paths;

                % Select first state
                results(i).S(t, 1) = 1;
                s                  = 1;

                % Select first action
                a                  = softmaxaction(beta.*Q(s,:));
                results(i).A(t, 1) = a;

                % State transition
                s2                 = binornd(1, taskparams.ptrans(a)) + 2;
                results(i).S(t, 2) = s2;

                % Select second action
                a2                 = softmaxaction(beta.*Q(s2,:));
                results(i).A(t, 2) = a2;

                % Receive feedback
                rprob           = reshape(paths, [2,2]);
                r               = binornd(1, rprob(s2-1, a2));
                results(i).R(t) = r;

                % Learn state transitions
                TUpdate     = alpha*(1-T(s, a, s2));
                T           = (1-alpha)*T;
                T(s, a, s2) = T(s, a, s2) + TUpdate;

                % Learn rewards
                rpe         = rho*r - Qf(s2, a2); % Reward prediction error
                %rpe         = r - Qf(s2, a2);
                results(i).rpe(t) = rpe;

                Qf(s, a)    = Qf(s, a) + alpha*(Qf(s2, a2) - Qf(s, a));
                Qf(s2, a2)  = Qf(s2, a2) + alpha*rpe;
                Qf(s, a)    = Qf(s, a) + alpha*rpe*lambda;

                for action = 1:2
                    Qmb(1, action) = T(s, action, 2)*max(Qf(2,:)) + T(s, action, 3)*max(Qf(3,:));
                end
                %Qmb(1, :) = [0.7, 0.3].*max(Qf(2,:)) + [0.3, 0.7].*max(Qf(3,:));

                Q = omega.*Qmb + (1-omega).*Qf;

                % Update paths
                paths = max(min(paths + taskparams.pathsigma*randn(1,4)*dt,0.75),0.25);
            end
        end
    end

    function data = writetable(results)
        nsubjects = size(results, 2);
        ntrials   = length(results(1).S);
        D         = nan(nsubjects*ntrials, 10);
        for i = 1:nsubjects
            starti = ((i-1)*ntrials)+1;
            endi   = ((i-1)*ntrials)+ntrials;

            D(starti:endi, 1)   = i;
            D(starti:endi, 2)   = 1:ntrials;
            D(starti:endi, 3:4) = results(i).S;
            D(starti:endi, 5:6) = results(i).A;
            D(starti:endi, 7)   = results(i).R;
            D(starti:endi, 8)   = results(i).rpe;
            D(starti+1:endi, 9)   = D(starti:endi-1, 7); %last trial rewarded

            crtrans = zeros(ntrials,1);
            crtrans(D(starti:endi, 4) == 2 & D(starti:endi, 5) == 1) = 1;
            crtrans(D(starti:endi, 4) == 3 & D(starti:endi, 5) == 2) = 1;
            D(starti:endi, 10) = crtrans; %last trial common or rare
        end

        data.table = D;
    end

    end
end
