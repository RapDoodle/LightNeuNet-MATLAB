%% Environment setup
clear
clc

%% Load the MNIST dataset
load mnist_uint8;

Xtrain = double(Xtrain) / 255;
ytrain = double(ytrain);
Xtest = double(Xtest) / 255;
ytest = double(ytest);

%% Build the DBN
model = DBN();
model.add(BernoulliRBM(784, 512));
model.add(BernoulliRBM(512, 384));
model.add(BernoulliRBM(384, 256));

model.compile();

%% Train the DBN
options.epochs = 20;
options.batchsize = 10;
options.momentum = 0;
options.alpha = 0.1;
options.decay = 0.00001;
options.k = 1;

model.fit(Xtrain, options);

%% Evaluate
Xreconstruct = model.evaluate(Xtrain);
m = size(Xtrain, 2);
diff = (1/m) * sum(sum((Xtrain - Xreconstruct) .^ 2));

%% Covert to sequential model
hiddenoptions.activation = "leeoscillator";
hiddenoptions.usebias = true;
hiddenoptions.kernelinitializer = "random";

outoptions.activation = "softmax";
outoptions.usebias = true;
outoptions.kernelinitializer = "random";

outputlayer = OutputLayer(10, outoptions);

seqmodel = model.tosequential({0, hiddenoptions, hiddenoptions, hiddenoptions}, outputlayer);

%% Train
options.batchsize = 128;
options.epochs = 800;
options.learningrate = 0.00001;
options.lambd = 1;
options.loss = "crossentropy";

seqmodel.fit(Xtrain, ytrain, options);

%% Test
probs = seqmodel.predict(Xtest);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, (1:10)');
correct = find(all(pred == ytest));
accuracy = length(correct) / size(ytest, 2);
fprintf('Classification accuracy is %3.2f%%\n', accuracy * 100);

%% Results
% Epoch: 1600: Classification accuracy is 75.46%, loss: 10.148031
% Classification accuracy is 75.33%
