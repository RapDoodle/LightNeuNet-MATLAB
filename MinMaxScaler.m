classdef MinMaxScaler < handle
    % Min-max Scaler. The vector or matrix
    % is standardized using the formula
    %   X = (X - min) / (max - min)
    % where mu is the mean and sigma is the
    % standard deviation.
    
    properties
        mmin
        mmax
    end
    
    methods
        function scaler = MinMaxScaler()
        end
        
        function fit(scaler, X, dim)
            scaler.mmin = min(X, [], dim);
            scaler.mmax = max(X, [], dim);
        end
        
        function Y = transform(scaler, X)
            Y = (X - scaler.mmin) ./ (scaler.mmax - scaler.mmin);
        end
        
        function Y = fittransform(scaler, X, dim)
            scaler.fit(X, dim);
            Y = scaler.transform(X);
        end
    end
end

