function [ cost, grad ] = stackedAECost(theta, hiddenSize, netconfig, ...
                                              options, x_train, y_train, groundTruth)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter
lambda = options.lambda;
gamma  = options.gamma;

% We first extract the part which compute the softmax gradient
% softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

softmaxTheta = reshape(theta(1:hiddenSize*size(y_train,1)),size(y_train,1),hiddenSize);
y_train = unique(y_train','rows','stable');

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*size(y_train,1)+1:end), netconfig);

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

%% --------------------------- YOUR CODE HERE -----------------------------

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = x_train;

for layer = (1:depth)
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end

numClasses = size(y_train,1);
Alpha = y_train*softmaxTheta;
M = bsxfun(@minus, Alpha*a{end}, max(Alpha*a{end},[],1));
M = exp(M);
p = bsxfun(@rdivide,M,sum(M));

cost = -1/numClasses * groundTruth(:)' * log(p(:)) + lambda/2 * sum(softmaxTheta(:) .^ 2)+ gamma/2 * sum(Alpha(:).^2);
softmaxThetaGrad = -1/numClasses * y_train'* (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta + gamma * y_train' *  Alpha;

d = cell(depth+1);

d{depth+1} = -(Alpha' * (groundTruth - p)) .* a{depth+1}.*(1-a{depth+1});

for layer = (depth:-1:2)
  d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer});
end

for layer = (depth:-1:1)
  stackgrad{layer}.w = (1/numClasses) * d{layer+1} * a{layer}';
  stackgrad{layer}.b = (1/numClasses) * sum(d{layer+1}, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end