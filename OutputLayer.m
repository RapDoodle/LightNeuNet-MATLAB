classdef OutputLayer < WeightedLayer
    %OUTPUTLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        y
    end
    
    methods
        function output = OutputLayer(units, options)
            %OUTPUTLAYER Construct an instance of this class
            %   Detailed explanation goes here
            output.units = units;
            output.activation = options.activation;
            output.usebias = options.usebias;
            output.kernelinitializer = options.kernelinitializer;
            output.options = options;
        end
        
        function y = forward(output, X, cache)
            y = forward@WeightedLayer(output, X, cache);
        end
        
        function backward(output, m, lambd)
            assert(m == size(output.y, 2), 'Inconsistency');
            
            if strcmp(output.activation, 'sigmoid') ...
                    || strcmp(output.activation, 'softmax') ...
                    || strcmp(output.activation, 'linear') ...
                    || strcmp(output.activation, 'tanh') ...
                    || strcmp(output.activation, 'leeoscillator')
                output.dZ = output.A - output.y;
                output.dW = (1/m) .* (output.dZ * output.prevlayer.A');
                output.db = (1/m) .* sum(output.dZ, 2);
            else
                throw(MException('OutputLayer:notImplemented', ...
                    'Not implemented activation function for backpropagation.'));
            end
            
            output.prevlayer.backward(m, lambd);
        end
        
        function newlayer = copy(layer)
            newlayer = OutputLayer(layer.units, layer.options);
            layer.move(newlayer);
        end
        
        function newlayer = move(layer, newlayer)
            move@WeightedLayer(layer, newlayer);
            newlayer.y = layer.y;
        end
    end
end

