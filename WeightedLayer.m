classdef WeightedLayer < Layer
    %WEIGHTEDLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        usebias
        activation
        kernelinitializer
        
        A
        dA
        
        Z
        dZ
        
        W
        dW
        
        b
        db
        
        vdW
        vdb
        
        sdW
        sdb
        
        options
    end
    
    methods(Abstract)
    end
    
    methods
        function init(layer, prevlayer)
            if ~isa(prevlayer, 'Layer')
                throw(MException('layer:noPreviousLayer', ...
                    'A previous layer must be defined for a layer layer'));
            end
            
            layer.b = zeros(layer.units, 1);
            
            if strcmp(layer.kernelinitializer, 'uniform')
                layer.W = -1 + rand(layer.units, prevlayer.units) * 2;
            
            elseif strcmp(layer.kernelinitializer, 'he')
                layer.W = randn(layer.units, prevlayer.units) * sqrt(2 / prevlayer.units);
            
            elseif strcmp(layer.kernelinitializer, 'random')
                % layer.W = -1 + rand(layer.units, prevlayer.units) * 2;
                epsilon_init = 0.12;
                layer.W = rand(layer.units, prevlayer.units) * 2 * epsilon_init - epsilon_init;
            
            else
                throw(MException('layer:unknownInitializer', ...
                    'Unknown initializer %s', layer.initializer));
            end
            
            % Default for Adam
            layer.vdW = 0;
            layer.vdb = 0;
            layer.sdW = 0;
            layer.sdb = 0;
            
            layer.prevlayer = prevlayer;
        end
        
        function y = forward(layer, X, cache)
            Zlocal = layer.W * X;
            
            if layer.usebias
                Zlocal = Zlocal + layer.b;
            end
            
            if strcmp(layer.activation, 'sigmoid')
                Alocal = sigmoid(Zlocal);
                
            elseif strcmp(layer.activation, 'relu')
                Alocal = relu(Zlocal);
                
            elseif strcmp(layer.activation, 'softmax')
                Alocal = softmax(Zlocal);
                
            elseif strcmp(layer.activation, 'tanh')
                Alocal = tanh(Zlocal);
                
            elseif strcmp(layer.activation, 'linear')
                Alocal = Zlocal;
                
            elseif strcmp(layer.activation, 'leeoscillator')
                Alocal = leeoscillator(Zlocal);
                
            else
                throw(MException('layer:unknownActivation', ...
                    'Unknown activation function %s', layer.activation));
            end
            
            if cache
                layer.Z = Zlocal;
                layer.A = Alocal;
            end
            
            y = Alocal;
        end
        
        function updateadam(layer, learningrate, beta1, beta2, epsilon, t)
            % Update the weights of the weighted layer
            % learningrate: the learning rate for each gradient descent
            % beta1: hyperparameter for Momentum, recommend: 0.9
            % beta2: hyperparameter for RMSProp, recommend: 0.999
            % epsilon: recommend: 0.00000001
            % t: current iteration
            
            % Momentum
            layer.vdW = beta1 * layer.vdW + (1-beta1) * layer.dW;
            layer.vdb = beta1 * layer.vdb + (1-beta1) * layer.db;
            
            % RMSProp
            layer.sdW = beta2 * layer.sdW + (1-beta2) * (layer.dW .^ 2);
            layer.sdb = beta2 * layer.sdb + (1-beta2) * (layer.db .^ 2);
            
            % Bias correction
            vdWcorrected = layer.vdW / (1-(beta1^t));
            vdbcorrected = layer.vdb / (1-(beta1^t));
            sdWcorrected = layer.sdW / (1-(beta2^t));
            sdbcorrected = layer.sdb / (1-(beta2^t));
            
            layer.W = layer.W - learningrate * ...
                (vdWcorrected ./ (sqrt(sdWcorrected) + epsilon));
            layer.b = layer.b - learningrate * ...
                (vdbcorrected ./ (sqrt(sdbcorrected) + epsilon));
            
            if isa(layer.prevlayer, 'WeightedLayer')
                layer.prevlayer.updateadam(...
                    learningrate, beta1, beta2, epsilon, t);
            end
        end
        
        function update(layer, learningrate)
            layer.W = layer.W - learningrate * layer.dW;
            layer.b = layer.b - learningrate * layer.db;
            
            if isa(layer.prevlayer, 'WeightedLayer')
                layer.prevlayer.update(learningrate);
            end
        end
        
        function newlayer = move(layer, newlayer)
            newlayer.usebias = layer.usebias;
            newlayer.activation = layer.activation;
            newlayer.kernelinitializer = layer.kernelinitializer;
            
            newlayer.A = layer.A;
            newlayer.dA = layer.dA;
            
            newlayer.Z = layer.Z;
            newlayer.dZ = layer.dZ;
            
            newlayer.W = layer.W;
            newlayer.dW = layer.dW;
            
            newlayer.b = layer.b;
            newlayer.db = layer.db;
        end
    end
end

