%===============================================================================
% RLPARAM Provides some functions that generate preset parameters for
%   reinforcement learning model objects in the rlfit-kit
%===============================================================================

classdef rlparam
    methods (Static)

        function [lr, beta, lambda, rho, w] = presetparams()
            lr     = learningrate();
            beta   = inversetemp();
            lambda = etraceparam();
            rho    = rewardsensitivity();
            w      = mbmfbalance();
        end

        function lr = learningrate(name, rng)
            lr = struct();

            if nargin == 1
                lr.name = name;
                lr.rng  = 'unit';
            elseif nargin == 2
                lr.name = name;
                lr.rng  = rng;
            else
                lr.name = 'Learning Rate (\alpha)';
                lr.rng  = 'unit';
            end
        end

        function beta = inversetemp(name, rng)
            beta = struct();

            if nargin == 1
              beta.name = name;
              beta.rng  = 'pos';
            elseif nargin == 2
              beta.name = name;
              beta.rng  = rng;
            else
              beta.name = 'Inverse Temperature (\beta)';
              beta.rng  = 'pos';
            end
        end

        function lambda = etraceparam(name, rng)
            lambda = struct();

            if nargin == 1
              lambda.name = name;
              lambda.rng  = 'unit';
            elseif nargin == 2
              lambda.name = name;
              lambda.rng  = rng;
            else
              lambda.name = 'Eligibility Trace Parameter (\lambda)';
              lambda.rng  = 'unit';
            end
        end

        function w = mbmfbalance(name, rng)
            w = struct();

            if nargin == 1
              w.name = name;
              w.rng  = 'unit';
            elseif nargin == 2
              w.name = name;
              w.rng  = rng;
            else
              w.name = 'MB/MF Balance (\omega)';
              w.rng  = 'unit';
            end
        end

        function rho = rewardsensitivity(name, rng)
            rho = struct();

            if nargin == 1
              rho.name = name;
              rho.rng  = 'unit';
            elseif nargin == 2
              rho.name = name;
              rho.rng  = rng;
            else
              rho.name = 'Reward Sensitivity (\rho)';
              rho.rng  = 'unit';
            end
        end
    end
end
