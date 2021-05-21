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
model = SequentialModel();

model.add(InputLayer(784));

options.activation = "tanh";
options.usebias = true;
options.kernelinitializer = "random";

model.add(DenseLayer(512, options));
model.add(DenseLayer(384, options));
model.add(DenseLayer(256, options));

options.activation = "softmax";
options.usebias = true;
options.kernelinitializer = "random";

model.add(OutputLayer(10, options));

% Compile the model
model.compile();

%% Train
options.batchsize = 128;
options.epochs = 20;
options.learningrate = 0.0001;
options.lambd = 1;
options.loss = "crossentropy";

model.fit(Xtrain, ytrain, options);

%% Test
probs = model.predict(Xtest);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, (1:10)');
correct = find(all(pred == ytest));
accuracy = length(correct) / size(ytest, 2);
fprintf('Classification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Max: Epoch: 20: Classification accuracy is 75.14%, loss: 10.042690
% Classification accuracy is 75.38%
