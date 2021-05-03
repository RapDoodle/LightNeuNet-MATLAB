%% Setting global parameters
clear
clc
use_gpu = false;

%% Load the MNIST dataset
load mnist_uint8;

X_train = double(X_train) / 255;
y_train = double(y_train);
X_test = double(X_test) / 255;
y_test = double(y_test);

if use_gpu
    X_train = gpuArray(X_train);
    y_train = gpuArray(y_train);
    X_test = gpuArray(X_test);
    y_test = gpuArray(y_test);
end

%% Model
input = InputLayer(784);

options.activation = "tanh";
options.use_bias = true;
options.kernel_initializer = "random";

hidden1 = DenseLayer(512, options);
hidden2 = DenseLayer(384, options);
hidden3 = DenseLayer(256, options);

options.activation = "softmax";
options.use_bias = true;
options.kernel_initializer = "random";

output = OutputLayer(10, options);

% Compile the model

input.next_layer = hidden1;

hidden1.init(input);
hidden1.next_layer = hidden2;

hidden2.init(hidden1);
hidden2.next_layer = hidden3;

hidden3.init(hidden2);
hidden3.next_layer = output;

output.init(hidden3);

%% Train
idx = 1;
batch = 128;
epoch = 20;
m = size(X_train, 2);

lastmsglen = 0;

for i = 1:epoch
    randpos = randperm(m);
    fprintf('\nEpoch: %d / %d\n', i, epoch);
    
    while idx <= m
        batchX = zeros(size(X_train, 1), batch);
        batchy = zeros(size(y_train,1), batch);
            
        for k = idx:min(idx + batch - 1, m)
            batchX(:, k - idx + 1) = X_train(:, randpos(k));
            batchy(:, k - idx + 1) = y_train(:, randpos(k));
        end
        
        % Forward propagation
        input.forward(batchX);

        % Backward propagation
        output.y = batchy;
        output.backward(size(output.y, 2), 1);
        output.update(0.0001);
        
        idx = idx + batch;
    end
    
    % disp(output.A(1, 1));
    probs = input.forward(X_train);
    [~, y] = max(probs, [], 1);
    pred = bsxfun(@eq, y, [1:10]');
    correct = find(all(pred == y_train));
    accuracy = length(correct) / size(y_train, 2);
    
    J = (1/m) * sum(sum((-y_train) .* log(probs) - (1-y_train) .* log(1-probs)));
    fprintf('Epoch: %d: Classification accuracy is %3.2f%%, loss: %f\n', i, accuracy * 100, J);
    
    idx = 1;
end

%% Test
probs = input.forward(X_test);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == y_test));
accuracy = length(correct) / size(y_test, 2);
fprintf('Classification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Max: Epoch: 20: Classification accuracy is 75.14%, loss: 10.042690
% Classification accuracy is 75.38%
