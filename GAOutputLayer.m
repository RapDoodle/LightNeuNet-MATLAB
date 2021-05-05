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
    end
end

