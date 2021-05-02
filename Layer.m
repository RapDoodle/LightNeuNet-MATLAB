classdef Layer < matlab.mixin.Heterogeneous & handle
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        units
        type
        
        next_layer = 0
        prev_layer = 0
    end
    
    methods(Abstract)
        y = forward(layer, X)
        backward(layer, m, lambd)
    end
end

