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
        y = forward(layer, X, cache)
        backward(layer, m, lambd)
        newlayer = copy(newlayer)
        newlayer = move(layer, mewlayer)
    end
end

