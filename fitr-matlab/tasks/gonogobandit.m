%===============================================================================
% INPUTS:
%       subjects = structure with subject specs
%           .N          = number of subjects (integer)
%           .K          = number of parameters (integer)
%           .paramNames = {1 by K} cell array of parameter name strings
%           .params     = [N by K] array of subject parameters (K params)
%
%
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada
%===============================================================================

classdef gonogobandit
    methods (Static)

        function results = vanilla(subjects, taskparams)

            %Initialize output structure
            results = struct();

            for i = 1:subjects.N
                alpha = subjects.params(i, 1);
                beta  = subjects.params(i, 2);
                Q     = zeros(taskparams.nstates, taskparams.nactions);
                results(i).S = mnrandix(1, taskparams.ptrans, taskparams.ntrials);
                for t = 1:taskparams.ntrials
                    s = results(i).S(t);

                    % Observe and Act
                    a = softmaxaction(beta.*Q(s, :));

                    % Receive feedback
                    r = binornd(1, taskparams.preward(s, a));%.*rewards(s, a);

                    % Learn
                    rpe = r - Q(s, a);
                    results(i).rpe(t) = rpe;
                    Q(s, a) = Q(s, a) + alpha*rpe;

                    %Store values
                    results(i).A(t, 1) = a;
                    results(i).R(t, 1) = r;
                end
            end
            disp(['----- Completed Simulation of ', num2str(subjects.N), ' Subjects -----']);
        end

        function results = rho(subjects, taskparams)
            % With reward sensitivity parameter

            %Initialize output structure
            results = struct();

            for i = 1:subjects.N
                alpha = subjects.params(i, 1);
                beta  = subjects.params(i, 2);
                rho   = subjects.params(i, 3);
                Q     = zeros(nstates, nactions);
                results(i).S = mnrandix(1, taskparams.ptrans, taskparams.ntrials);
                for t = 1:taskparams.ntrials
                    s = results(i).S(t);

                    % Observe and Act
                    a = softmaxaction(beta.*Q(s, :));

                    % Receive feedback
                    r = binornd(1, taskparams.preward(s, a)).*rewards(s, a);

                    % Learn
                    Q(s, a) = Q(s, a) + alpha*(rho*r - Q(s, a));

                    %Store values
                    results(i).A(t, 1) = a;
                    results(i).R(t, 1) = r;
                end
            end
        end

        function results = decay(subjects)
            % With Q value decay

            %Initialize output structure
            results = struct();

            for i = 1:subjects.N
                alpha = subjects.params(i, 1);
                beta  = subjects.params(i, 2);
                Q     = zeros(taskparams.nstates, taskparams.nactions);
                results(i).S = mnrandix(1, taskparams.ptrans, taskparams.ntrials);
                for t = 1:taskparams.ntrials
                    s = results(i).S(t);

                    % Observe and Act
                    a = softmaxaction(Q(s, :), beta);

                    % Receive feedback
                    r = binornd(1, taskparams.preward(s, a)).*rewards(s, a);

                    % Learn
                    Q(s, a) = Q(s, a) + alpha*(r - Q(s, a));

                    %Store values
                    results(i).A(t, 1) = a;
                    results(i).R(t, 1) = r;
                end
            end
        end

    end
end
