classdef Layer < matlab.mixin.Heterogeneous & handle
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        units
        type
        
        nextlayer = 0
        prevlayer = 0
    end
    
    methods(Abstract)
        y = forward(layer, X)
        backward(layer, m, lambd)
    end
end

