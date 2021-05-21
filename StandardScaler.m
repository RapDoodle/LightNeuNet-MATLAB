classdef StandardScaler < handle
    % Standard Scaler. The vector or matrix
    % is standardized using the formula
    %   X = (X - mu) / sigma
    % where mu is the mean and sigma is the
    % standard deviation.
    
    properties
        mmean
        mstd
    end
    
    methods
        function scaler = StandardScaler()
        end
        
        function fit(scaler, X, dim)
            scaler.mmean = mean(X, dim);
            scaler.mstd = std(X, 1, dim);
        end
        
        function Y = transform(scaler, X)
            Y = (X - scaler.mmean) ./ scaler.mstd;
        end
        
        function Y = fittransform(scaler, X, dim)
            scaler.fit(X, dim);
            Y = scaler.transform(X);
        end
    end
end

