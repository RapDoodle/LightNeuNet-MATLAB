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

options.activation = "sigmoid";
options.usebias = true;
options.kernelinitializer = "he";

model.add(GADenseLayer(512, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(384, options));
model.add(GADenseLayer(256, options));

options.activation = "softmax";
options.usebias = true;
options.kernelinitializer = "he";

model.add(GAOutputLayer(10, options));

%% Populate
model.populate(50);
% test = model.newindividual()

%% GA Optimization
options.keeprate = 0.6;
options.mutationrate = 0.01;
options.generations = 1000;
[minfitnesses, maxfitnesses] = model.run(@forwardpred, Xtrain, ytrain, options);

%% Introduce variations from pre-trained model
options.mutationrate = 0.05;
gamodel.mutateall(options);

%% Optimize with GA
options.keeprate = 0.6;
options.mutationrate = 0.01;
options.generations = 300;
[minfitnesses, maxfitnesses] = gamodel.run(@forwardpred, Xtrain, ytrain, options);

%% Test
probs = model.forest{1}.predict(Xtest);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == ytest));
accuracy = length(correct) / size(ytest, 2);
fprintf('\nClassification accuracy on test set is %3.2f%%\n', accuracy * 100);

%% Results
% [Generation [1000 / 3000] Fitness: 9.40 / 26.70]
% Classification accuracy on test set is 26.82%
