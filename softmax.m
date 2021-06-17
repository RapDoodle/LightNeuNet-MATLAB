function a = softmax(z)
% z should have be an (m * n) matrix.
a = exp(z) ./ sum(exp(z), (1));
end

