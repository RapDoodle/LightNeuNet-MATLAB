function X = gibbssample(P)
    X = double(sigmoid(P) > rand(size(P)));
end

