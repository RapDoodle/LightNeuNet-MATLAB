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

%% Build the DBN
model = DBN();
model.add(BernoulliRBM(784, 512));
model.add(BernoulliRBM(512, 384));
model.add(BernoulliRBM(384, 256));

model.compile();

%% Train the DBN
options.epochs = 1;
options.batch = 10;
options.momentum = 0;
options.alpha = 0.1;
options.decay = 0.00001;
options.k = 1;

model.fit(X_train, options);

%% Evaluate
Xreconstruct = model.evaluate(X_train);
m = size(X_train, 2);
diff = (1/m) * sum(sum((X_train - Xreconstruct) .^ 2));
