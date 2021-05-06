classdef GASequentialModel < SequentialModel
    %GASEQUENTIALMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        fitness = 0
    end
    
    methods
        function gamodel = GASequentialModel()
            gamodel = gamodel@SequentialModel();
        end
    end
end

