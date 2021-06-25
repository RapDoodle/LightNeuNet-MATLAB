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
options.epochs = 10;
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
hiddenoptions.activation = "sigmoid";
hiddenoptions.usebias = true;
hiddenoptions.kernelinitializer = "random";

outoptions.activation = "softmax";
outoptions.usebias = true;
outoptions.kernelinitializer = "random";

outputlayer = OutputLayer(10, outoptions);

seqmodel = model.tosequential({0, hiddenoptions, hiddenoptions, hiddenoptions}, outputlayer);

%% Convert to GA model
gamodel = seqmodel.togamodel();

%% Populate
gamodel.populate(50);
gamodel.replicatefrommodel(seqmodel);

%% Introduce variations from pre-trained model
options.mutationrate = 0.05;
gamodel.mutateall(options);

%% Optimize with GA
options.keeprate = 0.25;
options.mutationrate = 0.03;
options.generations = 500;
[minfitnesses, maxfitnesses] = gamodel.run(@forwardpred, Xtrain, ytrain, options);

%% Test
probs = gamodel.forest{1}.predict(Xtest);
[~, y] = max(probs, [], 1);
pred = bsxfun(@eq, y, [1:10]');
correct = find(all(pred == ytest));
accuracy = length(correct) / size(ytest, 2);
fprintf('\nClassification accuracy on test set is %3.2f%%\n', accuracy * 100);

%% Results
% [Generation [500 / 500] Fitness: 68.03 / 76.50]
% Classification accuracy on test set is 76.15%
