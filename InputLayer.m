classdef InputLayer < Layer
    %INPUTLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        A
    end
    
    methods
        function input = InputLayer(units)
            %INPUTLAYER Construct an instance of this class
            %   Detailed explanation goes here
            input.units = units;
        end
        
        function input = init(input, nextlayer)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            input.prev_layer = 0;
            input.nextlayer = nextlayer;
        end
        
        function y = forward(input, X, cache)
            input.A = X;
            y = input.nextlayer.forward(X, cache);
        end
        
        function backward(~, ~, ~)
        end
        
        function newlayer = move(input, newlayer)
            newlayer.A = input.A;
        end
        
        function newlayer = copy(input)
            newlayer = InputLayer(input.units);
            input.move(newlayer);
        end
    end
end

