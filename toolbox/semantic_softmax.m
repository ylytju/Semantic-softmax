function [softmaxModel] = semantic_softmax(lambda,gamma,x_train,y_train,yx,gt_label,options)
% softmanxTrain  Train a softmax model with the given parameters on the
% given data. Returns SoftmaxOptTheta, a vector contains the trained
% parameters for the model.

if ~exist('options','var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 100;
end

theta = 0.005 * randn(size(x_train,1)*size(y_train,1),1);
addpath minFunc
options.Method = 'lbfgs';

[c_transferOptTheta, ~] = minFunc(@(p) semantic_softmaxcost(p,lambda,gamma,x_train,y_train,yx,gt_label),theta,options);
softmaxModel.optTheta = reshape(c_transferOptTheta,size(y_train,1),size(x_train,1));

end