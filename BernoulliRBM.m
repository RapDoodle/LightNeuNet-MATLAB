classdef BernoulliRBM < matlab.mixin.Heterogeneous & handle
    %BERNOULLIRBM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        options
        visibleunits
        hiddenunits
        
        nextrbm = 0
        prevrbm = 0
        
        W
        vW

        b
        vb

        c
        vc
    end
    
    methods
        function rbm = BernoulliRBM(visibleunits, hiddenunits)
            rbm.visibleunits = visibleunits;
            rbm.hiddenunits = hiddenunits;
        end
        
        function init(rbm)
            
            rbm.W = zeros(rbm.hiddenunits, rbm.visibleunits);
            rbm.vW = zeros(rbm.hiddenunits, rbm.visibleunits);
            
            rbm.b = zeros(rbm.visibleunits, 1);
            rbm.vb = zeros(rbm.visibleunits, 1);
            
            rbm.c = zeros(rbm.hiddenunits, 1);
            rbm.vc = zeros(rbm.hiddenunits, 1);
        end
        
        function fit(rbm, X, options)
            idx = 1;
            m = size(X, 2);
            for i = 1:options.epochs
                randpos = randperm(m);
                fprintf('[BernoulliRBM] Epoch: %d / %d\n', i, options.epochs);

                while idx <= m
                    endidx = min(idx + options.batch - 1, m);
                    currentbatchsize = endidx - idx + 1;
                    v1 = zeros(size(X, 1), currentbatchsize);

                    for k = idx:endidx
                        v1(:, k - idx + 1) = X(:, randpos(k));
                    end
                    h1 = gibbssample(repmat(rbm.c, 1, currentbatchsize) + rbm.W * v1);

                    % Algorithm for RBM starts here
                    
                    % Initialze the chain that goes back and forth
                    if i == 1 && idx == 1
                       h2 = h1; 
                    end
                    
                    % k steps of gibbs sampling for the negative phase
                    for j = 1:options.k
                        v2 = gibbssample(repmat(rbm.b, 1, currentbatchsize) + rbm.W' * h2);
                        h2 = gibbssample(repmat(rbm.c, 1, currentbatchsize) + rbm.W * v2);
                    end

                    c1 = h1 * v1';
                    c2 = h2 * v2';
                    
                    rbm.vW = options.momentum * rbm.vW + ...
                        options.alpha * (c1 - c2 - options.decay * rbm.W) / currentbatchsize;
                    rbm.vb = options.momentum * rbm.vb + ...
                        options.alpha * (sum(v1' - v2')' - options.decay * rbm.b) / currentbatchsize;
                    rbm.vc = options.momentum * rbm.vc + ...
                        options.alpha * (sum(h1' - h2')' - options.decay * rbm.c) / currentbatchsize;

                    rbm.W = rbm.W + rbm.vW;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;
                    
                    % Algorithm for RBM ends here

                    idx = idx + currentbatchsize;
                end
                
                idx = 1;
            end
        end
        
        function X = rbmup(rbm, X)
            X = sigmoid(repmat(rbm.c, 1, size(X, 2)) + rbm.W * X);
        end
        
        function X = rbmdown(rbm, X)
            X = sigmoid(repmat(rbm.b, 1, size(X, 2)) + rbm.W' * X);
        end
    end
end

