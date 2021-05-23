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
options.keeprate = 0.6;
options.mutationrate = 0.01;
options.generations = 300;
[minfitnesses, maxfitnesses] = gamodel.run(@forwardpred, Xtrain, ytrain, options);

%% Results
% [Generation 300 / 300] Fitness: [91.72% / 92.49%]
