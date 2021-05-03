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
for i = 1:10000
    probs = input.forward(X_train);
    
    [~, y] = max(probs, [], 1);
    pred = bsxfun(@eq, y, [1:10]');
    correct = find(all(pred == y_train));
    accuracy = length(correct) / size(y_train, 2);
    
    m = size(output.y, 2);
    J = (1/m) * sum(sum((-y_train) .* log(probs) - (1-y_train) .* log(1-probs)));
    fprintf('Epoch: %d: Classification accuracy is %3.2f%%, loss: %f\n', i, accuracy * 100, J);
    
    output.y = y_train;
    
    
    % disp(output.A(1, 1));
    % disp("Loss: " + gather(loss));
    output.backward(size(output.y, 2), 0.1);
    output.update(0.02);
end

%% Test
probs = input.forward(X_test);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == y_test));
accuracy = length(correct) / size(y_test, 2);
fprintf('Classification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Max: Epoch: 1043: Classification accuracy is 75.14%, loss: 10.042690
% Classification accuracy is 75.38%
