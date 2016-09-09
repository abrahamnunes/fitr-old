classdef threearmedbandit
    methods (Static)
        function results = vanilla(subjects)

            nsubjects = subjects.N;
            ntrials   = 50;
            nstates   = 2; ptrans = [0.5 0.5];
            nactions  = 3;
            Q         = zeros(nstates, nactions, nsubjects);

            rprob = round([0.6 0.56 0.2; 0.77 0.25 0.44; 0.43 0.14 0.16]);

            for i = 1:nsubjects
                alpha = subjects.params(i, 1);
                beta  = subjects.params(i, 2);
                for t = 1:ntrials
                    s = mnrandix(1, ptrans, 1);
                    results(i).s(t) = s;
                    a = softmaxaction(Q(s,:, i), beta);
                    results(i).a(t) = a;

                    r = binornd(1, rprob(s, a));
                    results(i).r(t) = r;
                    rpe = r - Q(s, a, i);
                    results(i).rpe(t) = rpe;
                    Q(:,:,i)   = (1-alpha)*Q(:,:,i);
                    Q(s, a, i) = Q(s, a, i) + alpha*rpe;
                end
            end
        end
    end
end
