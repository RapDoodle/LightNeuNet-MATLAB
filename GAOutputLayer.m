classdef GAOutputLayer < GAWeightedLayer & OutputLayer
    %GAOUTPUTLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function gaoutput = GAOutputLayer(units, options)
            %INPUTLAYER Construct an instance of this class
            %   Detailed explanation goes here
            gaoutput = gaoutput@OutputLayer(units, options);
        end
        
        function newlayer = copy(layer)
            newlayer = GAOutputLayer(layer.units, layer.options);
            layer.move(newlayer);
        end
        
        function newlayer = move(layer, newlayer)
            move@OutputLayer(layer, newlayer);
        end
    end
end

