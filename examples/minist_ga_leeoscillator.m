%% Environment setup
clear
clc

%% Load the MNIST dataset
load mnist_uint8;

Xtrain = double(Xtrain) / 255;
ytrain = double(ytrain);
Xtest = double(Xtest) / 255;
ytest = double(ytest);

%% Model
model = GAModel();

model.add(GAInputLayer(784));

options.activation = "leeoscillator";
options.usebias = true;
options.kernelinitializer = "random";

model.add(GADenseLayer(512, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(256, options));

options.activation = "softmax";
options.usebias = true;
options.kernelinitializer = "random";

model.add(GAOutputLayer(10, options));

%% Populate
model.populate(50);

%% GA Optimization
options.keeprate = 0.6;
options.mutationrate = 0.01;
options.generations = 300;
[minfitnesses, maxfitnesses] = model.run(@forwardpred, Xtrain, ytrain, options);

%% Test
probs = model.forest{1000}.predict(Xtest);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == ytest));
accuracy = length(correct) / size(ytest, 2);
fprintf('\nClassification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Max: Epoch: 20: Classification accuracy is 75.14%, loss: 10.042690
% Classification accuracy is 75.38%
