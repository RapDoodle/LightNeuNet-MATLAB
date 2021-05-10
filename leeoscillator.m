function z = leeoscillator(x)
load('lee_oscillator.mat');
z = x(:);
m = size(Z, 2);
n = length(z);
for i = 1:n
    if z(i) < -1 || z(i) > 1
        z(i) = sigmoid(z(i));
    else
        row = int32((z(i)+1)/stepsize);
        col = int32(rand() * (m - 1)) + 1;
        z(i) = Z(row, col);
    end
end
z = reshape(z, size(x));
end