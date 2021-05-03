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

options.activation = "sigmoid";
options.use_bias = true;
options.kernel_initializer = "random";

hidden1 = DenseLayer(512, options);
hidden2 = DenseLayer(512, options);
hidden3 = DenseLayer(512, options);
hidden4 = DenseLayer(256, options);
hidden5 = DenseLayer(256, options);
hidden6 = DenseLayer(256, options);
hidden7 = DenseLayer(128, options);
hidden8 = DenseLayer(128, options);
hidden9 = DenseLayer(128, options);
hidden10 = DenseLayer(64, options);

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
hidden3.next_layer = hidden4;

hidden4.init(hidden3);
hidden4.next_layer = hidden5;

hidden5.init(hidden4);
hidden5.next_layer = hidden6;

hidden6.init(hidden5);
hidden6.next_layer = hidden7;

hidden7.init(hidden6);
hidden7.next_layer = hidden8;

hidden8.init(hidden7);
hidden8.next_layer = hidden9;

hidden9.init(hidden8);
hidden9.next_layer = hidden10;

hidden10.init(hidden9);
hidden10.next_layer = output;

output.init(hidden10);

%% Train
for i = 1:10000
    probs = input.forward(X_train);
    
    [~, y] = max(probs, [], 1);
    pred = bsxfun(@eq, y, [1:10]');
    correct = find(all(pred == y_train));
    accuracy = length(correct) / size(y_train, 2);
    fprintf('Epoch: %d: Classification accuracy is %3.2f%%\n', i, accuracy * 100);
    
    output.y = y_train;
    loss = (1/(2*size(output.y, 2)))*sum((y_train - probs) .^ 2, [1, 2]);
    disp(output.A(1, 1));
    disp("Loss: " + gather(loss));
    output.backward(size(output.y, 2), 0.1);
    output.update(0.05);
end

%% Test
probs = input.forward(X_test);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == y_test));
accuracy = length(correct) / size(y_test, 2);
fprintf('Classification accuracy is %3.2f%%\n', accuracy * 100);
