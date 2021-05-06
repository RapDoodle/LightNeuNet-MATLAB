classdef GAWeightedLayer < GALayer
    %GAWEIGHTEDLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function mutate(layer, mutationrate)
            % Get the probability of mutation for each weight
            Wmutopts = rand(size(layer.W, 1), size(layer.W, 2));
            bmutopts = rand(size(layer.b, 1), size(layer.b, 2));
            
            Wmutmask = zeros(size(layer.W, 1), size(layer.W, 2));
            bmutmask = zeros(size(layer.b, 1), size(layer.b, 2));
            
            % # Determine which weight to mutate
            Wmutmask(Wmutopts < mutationrate) = rand() - 0.5;
            bmutmask(bmutopts < mutationrate) = rand() - 0.5;

            % Update the weights
            layer.W = layer.W + Wmutmask;
            layer.b = layer.b + bmutmask;            
        end
        
        function crossover(layer, mateW, mateb)
            % Get the probability of crossover for each weight
            Wcoopts = rand(size(layer.W, 1), size(layer.W, 2));
            bcoopts = rand(size(layer.b, 1), size(layer.b, 2));
            
            Wcomask = zeros(size(layer.W, 1), size(layer.W, 2));
            bcomask = zeros(size(layer.b, 1), size(layer.b, 2));
            Wcomaskinv = zeros(size(layer.W, 1), size(layer.W, 2));
            bcomaskinv = zeros(size(layer.b, 1), size(layer.b, 2));
            
            % # Determine which weight to crossover
            Wcomask(Wcoopts >= 0.5) = 1;
            bcomask(bcoopts >= 0.5) = 1;
            Wcomaskinv(Wcoopts < 0.5) = 1;
            bcomaskinv(bcoopts < 0.5) = 1;

            % Apply the invert mask on layer.W and layer.b
            layer.W = layer.W .* Wcomaskinv;
            layer.b = layer.b .* bcomaskinv;

            % Apply the mask on mateW and mateb
            layer.W = layer.W + mateW .* Wcomask;
            layer.b = layer.b + mateb .* bcomask;
        end
    end
end

