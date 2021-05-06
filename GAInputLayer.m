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
        
        function mutate(~, ~)         
        end
        
        function crossover(~, ~, ~)
        end
        
        function newlayer = copy(layer)
            newlayer = GAInputLayer(layer.units);
            layer.move(newlayer);
        end
        
        function newlayer = move(layer, newlayer)
            move@InputLayer(layer, newlayer);
        end
    end
end

