classdef DenseLayer < WeightedLayer
    %DENSE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function dense = DenseLayer(units, options)
            dense.type = LayerType.HiddenLayer;
            dense.units = units;
            dense.activation = options.activation;
            dense.usebias = options.usebias;
            dense.kernelinitializer = options.kernelinitializer;
            dense.options = options;
        end
        
        function setnextlayer(dense, nextlayer)
            if ~isa(nextlayer, 'Layer')
                throw(MException('Dense:notALayer', ...
                    'Not a layer.'));
            end
            dense.nextlayer = nextlayer;
        end
        
        function y = forward(dense, X, cache)
            Aout = forward@WeightedLayer(dense, X, cache);
            
            if ~isa(dense.nextlayer, 'Layer')
                throw(MException('Dense:notALayer', ...
                    'Not a layer.'));
            end
            
            % Pass the output of the current layer to the next layer
            % This y is not the y in the output layer, just the output
            % resulted in the current layer
            y = dense.nextlayer.forward(Aout, cache);
            
        end
        
        function backward(dense, m, lambd)
            dense.dA = dense.nextlayer.W' * dense.nextlayer.dZ;
            
            if strcmp(dense.activation, 'relu')
                dense.dZ = dense.dA .* (dense.dA > 0);
            
            elseif strcmp(dense.activation, 'linear')
                dense.dZ = dense.dA;
            
            elseif strcmp(dense.activation, 'sigmoid')
                s = sigmoid(dense.Z);
                dense.dZ = dense.dA .* s .* (1 - s);
                
            elseif strcmp(dense.activation, 'tanh')
                dense.dZ = (dense.nextlayer.W' * dense.nextlayer.dZ) .* ...
                    (1 - dense.A .^ 2);
                
            elseif strcmp(dense.activation, 'leeoscillator')
                dense.dZ = dense.dA .* leeoscillatorGradient(dense.Z);
                
            end
            
            dense.dW = (1/m) .* (dense.dZ * dense.prevlayer.A') + ...
                (lambd / m) .* dense.W;
            dense.db = (1/m) .* sum(dense.dZ, 2);
            
            dense.prevlayer.backward(m, lambd);
        end
        
        function newlayer = copy(layer)
            newlayer = DenseLayer(layer.units, layer.options);
            layer.move(newlayer);
        end
        
        function newlayer = move(layer, newlayer)
            move@WeightedLayer(layer, newlayer);
        end
        
    end
end

