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
            output.use_bias = options.use_bias;
            output.kernel_initializer = options.kernel_initializer;
        end
        
        function y = forward(output, X)
            forward@WeightedLayer(output, X);
            y = output.A;
        end
        
        function backward(output, m, lambd)
            assert(m == size(output.y, 2), 'Inconsistency');
            
            if strcmp(output.activation, 'sigmoid') || strcmp(output.activation, 'softmax')
                output.dZ = output.A - output.y;
                output.dW = (1/m) .* (output.dZ * output.prevlayer.A');
                output.db = (1/m) .* sum(output.dZ, 2);
            end
            
            output.prevlayer.backward(m, lambd);
        end
    end
end

