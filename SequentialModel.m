classdef SequentialModel < matlab.mixin.Heterogeneous & handle
    %SEQUENTIALMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inputlayer = 0
        outputlayer = 0
        layers = {}
        compiled = false
    end
    
    methods
        function model = SequentialModel()
        end
        
        function add(model, layer)
            if ~isa(layer, 'Layer')
                throw(MException('SequentialModel:notALayer', ...
                    'Not a layer.'));
            end
            model.layers{end+1} = layer;
        end
        
        function y = predict(model, X)
            y = model.inputlayer.forward(X);
        end
        
        function compile(model)
            if length(model.layers) <= 0
                throw(MException('SequentialModel:noLayers', ...
                    'The model does not contain any layer.'));
            end
            if ~isa(model.layers{1}, 'InputLayer')
                throw(MException('SequentialModel:invalidLayer', ...
                    'The first layer of the model is not an input layer.'));
            end
            if ~isa(model.layers{end}, 'OutputLayer')
                throw(MException('SequentialModel:invalidLayer', ...
                    'The last layer of the model is not an output layer.'));
            end
            
            for i = 1:length(model.layers)
                if i == 1
                    % The first layer
                    model.inputlayer = model.layers{1};
                else
                    % The hidden layers and output layer
                    model.layers{i}.init(model.layers{i-1});
                    model.layers{i-1}.next_layer = model.layers{i};
                    if i == length(model.layers)
                        model.outputlayer = model.layers{i};
                    end
                end
            end
            
            model.compiled = true;
            
        end
        
        function history = fit(model, X, y, options)
            % X should have a size of (n1, m)
            % y should have a size of (n2, m)
            history = zeros(options.epoch, 1);
            idx = 1;
            m = size(X, 2);
            for i = 1:options.epoch
                randpos = randperm(m);
                fprintf('\nEpoch: %d / %d\n', i, options.epoch);

                while idx <= m
                    batchX = zeros(size(X, 1), options.batch);
                    batchy = zeros(size(y, 1), options.batch);

                    for k = idx:min(idx + options.batch - 1, m)
                        batchX(:, k - idx + 1) = X(:, randpos(k));
                        batchy(:, k - idx + 1) = y(:, randpos(k));
                    end

                    % Forward propagation
                    model.inputlayer.forward(batchX);

                    % Backward propagation
                    model.outputlayer.y = batchy;
                    model.outputlayer.backward(size(model.outputlayer.y, 2), options.lambd);
                    model.outputlayer.update(options.learningrate);

                    idx = idx + options.batch;
                end

                % disp(output.A(1, 1));
                probs = model.inputlayer.forward(X);
                [~, y_out] = max(probs, [], 1);
                pred = bsxfun(@eq, y_out, [1:10]');
                correct = find(all(pred == y));
                accuracy = length(correct) / size(y, 2);
                
                if strcmp(options.loss, "crossentropy")
                    J = (1/m) * sum(sum((-y) .* log(probs) - (1-y) .* log(1-probs)));
                else
                    % Not found.
                    J = 0;
                end
                
                history(i) = J;
                
                fprintf('Epoch: %d: Classification accuracy is %3.2f%%, loss: %f\n', ...
                    i, accuracy * 100, J);

                idx = 1;
            end
        end
    end
end

