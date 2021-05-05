classdef GADenseLayer < GAWeightedLayer & DenseLayer
    %GADENSELAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function gadense = GADenseLayer(units, options)
            gadense = gadense@DenseLayer(units, options);
        end
    end
end

