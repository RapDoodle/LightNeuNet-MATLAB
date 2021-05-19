%% Setting global parameters
clear
clc
usegpu = false;

%% Load the MNIST dataset
load mnist_uint8;

Xtrain = double(X_train) / 255;
ytrain = double(y_train);
Xtest = double(X_test) / 255;
ytest = double(y_test);

if usegpu
    Xtrain = gpuArray(Xtrain);
    ytrain = gpuArray(ytrain);
    Xtest = gpuArray(Xtest);
    ytest = gpuArray(ytest);
end

clear X_train;
clear y_train;
clear X_test;
clear y_test;

%% Build the DBN
model = DBN();
model.add(BernoulliRBM(784, 512));
model.add(BernoulliRBM(512, 384));
model.add(BernoulliRBM(384, 256));

model.compile();

%% Train the DBN
options.epochs = 20;
options.batch = 10;
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
hiddenoptions.use_bias = true;
hiddenoptions.kernel_initializer = "random";

outoptions.activation = "softmax";
outoptions.use_bias = true;
outoptions.kernel_initializer = "random";

outputlayer = OutputLayer(10, outoptions);

seqmodel = model.tosequential({0, hiddenoptions, hiddenoptions, hiddenoptions}, outputlayer);

%% Train
options.batch = 128;
options.epoch = 800;
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
