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
            y = model.inputlayer.forward(X, false);
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
                    model.layers{i-1}.nextlayer = model.layers{i};
                    if i == length(model.layers)
                        model.outputlayer = model.layers{i};
                    end
                end
            end
            
            model.compiled = true;
            
        end
        
        function history = fit(model, X, y, options, verbose)
            % X should have a size of (n1, m)
            % y should have a size of (n2, m)
            % verbose can be ignored. By defulat: 1
            if nargin < 5
                verbose = 1;
            end
            
            history = zeros(options.epochs, 1);
            idx = 1;
            m = size(X, 2);
            for i = 1:options.epochs
                randpos = randperm(m);
                if verbose ~= 0
                    fprintf('\nEpoch: %d / %d\n', i, options.epochs);
                end

                while idx <= m
                    batchX = zeros(size(X, 1), options.batchsize);
                    batchy = zeros(size(y, 1), options.batchsize);

                    for k = idx:min(idx + options.batchsize - 1, m)
                        batchX(:, k - idx + 1) = X(:, randpos(k));
                        batchy(:, k - idx + 1) = y(:, randpos(k));
                    end

                    % Forward propagation
                    model.inputlayer.forward(batchX, true);

                    % Backward propagation
                    model.outputlayer.y = batchy;
                    model.outputlayer.backward(size(model.outputlayer.y, 2), options.lambd);
                    
                    % Has the field optimizer adn did not specified 'none'
                    if isfield(options, 'optimizer') && strcmp(options.optimizer, 'none') == 0
                       if strcmp(options.optimizer, 'adam')
                           model.outputlayer.updateadam( ...
                               options.learningrate, ...
                               options.beta1, ...
                               options.beta2, ...
                               options.epsilon, ...
                               i);
                       else
                           throw(MException('SequentialModel:unknownOptimizer', ...
                                'Unkown optimizer.'));
                       end
                    else
                        model.outputlayer.update(options.learningrate);
                    end

                    idx = idx + options.batchsize;
                end

                probs = model.inputlayer.forward(X, false);
                
                if strcmp(options.loss, "crossentropy") || strcmp(options.loss, "binarycrossentropy")
                    if strcmp(options.loss, "binarycrossentropy")
                        loss = (1/m) * sum(sum((-y) .* log(probs) - (1-y) .* log(1-probs)));
                    elseif strcmp(options.loss, "crossentropy")
                        loss = (1/m) * sum(sum((-y) .* log(probs)));
                    else
                        % This line of code should never be executed
                        loss = -1;
                    end
                    [~, yout] = max(probs, [], 1);
                    pred = bsxfun(@eq, yout, (1:size(y, 1))');
                    correct = find(all(pred == y));
                    accuracy = length(correct) / size(y, 2);
                    if verbose ~= 0
                        fprintf('Epoch: %d: Classification accuracy is %3.2f%%, loss: %f\n', ...
                            i, accuracy * 100, loss);
                    end
                elseif strcmp(options.loss, "mse") || strcmp(options.loss, "mae")
                    if strcmp(options.loss, "mse")
                        % Mean squared error
                        loss = (1/m) * sum(sum((y-probs).^2));
                    else
                        % Mean absolute error
                        loss = (1/m) * sum(sum(abs(y-probs)));
                    end
                    if verbose ~= 0
                        fprintf('Epoch: %d: loss: %f\n', i, loss);
                    end
                else
                    % Metric not found
                    throw(MException('SequentialModel:unknownMetric', ...
                        'The model does not have a valid metric.'));
                end
                
                history(i) = loss;

                idx = 1;
            end
        end
        
        function gamodel = togamodel(model)
            n = length(model.layers);
            assert(n > 0);
            
            gamodel = GAModel();
            % Input layer
            gamodel.add(GAInputLayer(model.layers{1}.units));
            
            % Hidden layers and output layers
            for i = 2:(n-1)
                gamodel.add(GADenseLayer(model.layers{i}.units, ...
                    model.layers{i}.options));
            end
            gamodel.add(GAOutputLayer(model.layers{n}.units, ...
                model.layers{n}.options));
            
            % Copy weights
            for i = 2:n
                gamodel.layersbp{i}.W = model.layers{i}.W;
                gamodel.layersbp{i}.b = model.layers{i}.b;
            end
        end
    end
end

