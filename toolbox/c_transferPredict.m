function [meanAcc] = c_transferPredict(theta,outputSize,hiddenSize, netconfig,test)

% c_transferModel: model trained using softmaxTrain
% x_test: the p * n matrix, where each column data (:,i) corresponds to a
% single test visual instance
% y_test: the q * N matrix, where each column data(:,i) corresponds to a
% class-level semantic vector
%

softmaxTheta = reshape(theta(1:hiddenSize*outputSize),outputSize,hiddenSize);
stack = params2stack(theta(hiddenSize*outputSize+1:end), netconfig);

x_test = test.x;
y_test = test.y;
labels = test.labels;
cla_num= test.cla_num;

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = x_test;

for layer = (1:depth)
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end
sim = y_test'*softmaxTheta*a{end};
meanAcc = accuracy(sim', labels, cla_num);
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end