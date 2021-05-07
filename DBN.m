classdef DBN < handle
    %DBN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inputlayer = 0
        layers = {}
        compiled = false
    end
    
    methods
        function dbn = DBN()
        end
        
        function add(dbn, layer)
            if ~isa(layer, 'BernoulliRBM')
                throw(MException('DBN:notAValidLayer', ...
                    'Not a valid layer.'));
            end
            dbn.layers{end+1} = layer;
        end
        
        function compile(dbn)
            if length(dbn.layers) <= 0
                throw(MException('DBN:noLayers', ...
                    'The model does not contain any layer.'));
            end
            
            for i = 1:length(dbn.layers)
                % Register the input layer
                if i == 1
                    dbn.inputlayer = dbn.layers{1};
                end
                
                % Check for the match of dimensions
                if i ~= length(dbn.layers)
                    assert(dbn.layers{i}.hiddenunits == dbn.layers{i+1}.visibleunits);
                end
                
                % The hidden layers and output layer
                dbn.layers{i}.init();
            end        
            
            dbn.compiled = true;
        end
        
        function fit(dbn, X, options)
            for i = 1:length(dbn.layers)
                dbn.layers{i}.fit(X, options);
                X = dbn.layers{i}.rbmup(X);
            end
        end
        
        function X = evaluate(dbn, X)
            % Forward pass
            for i = 1:length(dbn.layers)
                X = dbn.layers{i}.rbmup(X);
            end
            
            % Backward pass
            for i = length(dbn.layers):-1:1
                X = dbn.layers{i}.rbmdown(X);
            end
        end
    end
end

