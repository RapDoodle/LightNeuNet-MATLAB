classdef GALayer < handle
    %GALAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods(Abstract)
        mutate(layer, mutationrate)
        crossover(layer, mateW, mateb)
    end
end

