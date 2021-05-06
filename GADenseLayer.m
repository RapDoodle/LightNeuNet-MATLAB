classdef GADenseLayer < GAWeightedLayer & DenseLayer
    %GADENSELAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function gadense = GADenseLayer(units, options)
            gadense = gadense@DenseLayer(units, options);
        end
        
        function newlayer = copy(layer)
            newlayer = GADenseLayer(layer.units, layer.options);
            layer.move(newlayer);
        end
        
        function newlayer = move(layer, newlayer)
            move@DenseLayer(layer, newlayer);
        end
    end
end

