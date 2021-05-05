classdef GAInputLayer < InputLayer & GALayer
    %GAINPUTLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function gainput = GAInputLayer(units)
            %INPUTLAYER Construct an instance of this class
            %   Detailed explanation goes here
            gainput = gainput@InputLayer(units);
        end
        
        function mutate(layer, mutationrate)         
        end
        
        function crossover(layer, mateW, mateb)
        end
    end
end

