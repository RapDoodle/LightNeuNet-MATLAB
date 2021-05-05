classdef WeightedLayer < Layer
    %WEIGHTEDLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        use_bias
        activation
        kernel_initializer
        
        A
        dA
        
        Z
        dZ
        
        W
        dW
        
        b
        db
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
            
            if strcmp(layer.kernel_initializer, 'uniform')
                layer.W = -1 + rand(layer.units, prevlayer.units) * 2;
            
            elseif strcmp(layer.kernel_initializer, 'he')
                layer.W = randn(layer.units, prevlayer.units) * sqrt(2 / prevlayer.units);
            
            elseif strcmp(layer.kernel_initializer, 'random')
                % layer.W = -1 + rand(layer.units, prevlayer.units) * 2;
                epsilon_init = 0.12;
                layer.W = rand(layer.units, prevlayer.units) * 2 * epsilon_init - epsilon_init;
            
            else
                throw(MException('layer:unknownInitializer', ...
                    'Unknown initializer %s', layer.initializer));
            end
            
            layer.prevlayer = prevlayer;
        end
        
        function y = forward(layer, X)
            layer.Z = layer.W * X;
            
            if layer.use_bias
                layer.Z = layer.Z + layer.b;
            end
            
            if strcmp(layer.activation, 'sigmoid')
                layer.A = sigmoid(layer.Z);
                
            elseif strcmp(layer.activation, 'relu')
                layer.A = relu(layer.Z);
                
            elseif strcmp(layer.activation, 'softmax')
                layer.A = softmax(layer.Z);
                
            elseif strcmp(layer.activation, 'tanh')
                layer.A = tanh(layer.Z);
                
            elseif strcmp(layer.activation, 'linear')
                layer.A = layer.Z;
                
            else
                throw(MException('layer:unknownActivation', ...
                    'Unknown activation function %s', layer.activation));
            end
            
            y = [];
        end
        
        function update(layer, learning_rate)
            layer.W = layer.W - learning_rate * layer.dW;
            layer.b = layer.b - learning_rate * layer.db;
            
            if isa(layer.prevlayer, 'WeightedLayer')
                layer.prevlayer.update(learning_rate);
            end
        end
    end
end

