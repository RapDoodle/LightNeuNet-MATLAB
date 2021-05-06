classdef GAModel < handle
    %GAMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        forest = {}
        % Blueprint for the layers
        layersbp = {}
    end
    
    methods
        function gamodel = GAModel()
        end
        
        function add(gamodel, layer)
            if ~isa(layer, 'GALayer')
                throw(MException('GAModel:notALayer', ...
                    'Not a GALayer.'));
            end
            gamodel.layersbp{end+1} = layer;
        end
        
        function gaseqmodel = newindividual(gamodel)
            gaseqmodel = GASequentialModel();
            
            for i = 1:length(gamodel.layersbp)
                gaseqmodel.add(gamodel.layersbp{i}.copy());
            end
            
            gaseqmodel.compile();
            gamodel.forest{end+1} = gaseqmodel;
        end
        
        function populate(gamodel, n)
            % n is the size the of population
            for i = 1:n
                gamodel.newindividual();
            end
        end
        
        function [minfitnesses, maxfitnesses] = run(gamodel, func, X, y, options)
            nummodels = length(gamodel.forest);
            minfitnesses = zeros(options.generations, 1);
            maxfitnesses = zeros(options.generations, 1);
            
            for generation = 1:options.generations
                for i = 1:nummodels
                    if mod(i, 10) == 0 || i == 1
                        fprintf('[Generation %d / %d] Simulating: %d / %d\n', ...
                        generation, options.generations, i, nummodels);
                    end
                    func(gamodel.forest{i}, X, y);
                end

                gamodel.sortforest();
                minfitnesses(generation) = gamodel.forest{nummodels}.fitness;
                maxfitnesses(generation) = gamodel.forest{1}.fitness;
                
                fprintf('Fitness: [%3.2f%% / %3.2f%%]\n', ...
                    minfitnesses(generation), maxfitnesses(generation));

                % Natural selection
                keepidx = int32(nummodels * options.keeprate);
                gamodel.forest = {gamodel.forest{:, 1:keepidx}};

                fitnesses = cellfun(@(x)x.fitness, gamodel.forest);
                chances = fitnesses / sum(fitnesses);
                cp = [0, cumsum(chances)];

                % Reproduction
                for i = keepidx+1:nummodels
                    r = rand();
                    sela = find(r>cp, 1, 'last');
                    r = rand();
                    selb = find(r>cp, 1, 'last');
                    modela = gamodel.forest{sela};
                    modelb = gamodel.forest{selb};

                    newmodel = gamodel.newindividual();

                    % Set the parameters
                    for layeridx = 2:length(newmodel.layers)
                        newmodel.layers{layeridx}.W = modela.layers{layeridx}.W;
                        newmodel.layers{layeridx}.b = modela.layers{layeridx}.b;
                        newmodel.layers{layeridx}.crossover(modelb.layers{layeridx}.W, ...
                            modelb.layers{layeridx}.b);
                        newmodel.layers{layeridx}.mutate(options.mutationrate);
                    end
                end
                
                assert(nummodels == length(gamodel.forest));
            end
        end

        function sortforest(gamodel)
            fitnesses = cellfun(@(x)x.fitness, gamodel.forest);
            [~,sortidx] = sort(fitnesses, "descend");
            gamodel.forest = gamodel.forest(sortidx);
        end
    end
end
