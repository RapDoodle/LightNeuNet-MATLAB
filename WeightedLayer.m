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
        
        function update(layer, learning_rate)
            layer.W = layer.W - learning_rate * layer.dW;
            layer.b = layer.b - learning_rate * layer.db;
            
            if isa(layer.prevlayer, 'WeightedLayer')
                layer.prevlayer.update(learning_rate);
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

