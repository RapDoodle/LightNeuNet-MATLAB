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
model = GAModel();

model.add(GAInputLayer(784));

options.activation = "tanh";
options.use_bias = true;
options.kernel_initializer = "random";

model.add(GADenseLayer(512, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(256, options));

options.activation = "softmax";
options.use_bias = true;
options.kernel_initializer = "random";

model.add(GAOutputLayer(10, options));

%% GA Testing
model.populate(300);
% test = model.newindividual()

%% AA
options.keeprate = 0.6;
options.mutationrate = 0.01;
options.generations = 3;
[minfitnesses, maxfitnesses] = model.run(@forwardpred, X_train, y_train, options);

%% Just to make sure the model functions correctly
options.batch = 128;
options.epoch = 20;
options.learningrate = 0.0001;
options.lambd = 1;
options.loss = "crossentropy";

model.forest{1}.fit(X_train, y_train, options);



%% Test
probs = model.forest{1000}.predict(X_test);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == y_test));
accuracy = length(correct) / size(y_test, 2);
fprintf('\nClassification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Max: Epoch: 20: Classification accuracy is 75.14%, loss: 10.042690
% Classification accuracy is 75.38%
