
%%
% Semantic softmax loss for zero-shot learning
% Written by Yun-long Yu, yuyunlong@tju.edu.cn
% updated on May 22rd, 2017.
%%
clc;
clear;
close all;

addpath toolbox minFunc
Data = load('./fea/CUB_att.mat');
Train = Data.Train;
Test  = Data.Test;
%% STEP 1: 
x_train = normcol_equal(Train.im_fea);
y_train = normcol_equal(Train.semantic_fea);
gt_label = Train.GT_label;

%% 
x_test = normcol_equal(Test.im_fea);
y_test = normcol_equal(Test.semantic_uni);
labels = Test.index;
cla_num = Test.cla_num;

%% hyper-parameter setting
lambda = 5e-4;   %att 5e-3 word 1e0  CUB_att 5e-4 word 1e-3 SUN 1e-4
gamma = 5e-6;    %att 1e-4 word 1e-4 CUB_att 5e-6 word 1e-6 SUN 1e-8
%% Train 
disp('...Training...');
options = struct;
options.maxIter = 50;
yx = y_train*x_train';
softmaxModel = semantic_softmax(lambda,gamma,x_train,y_train,yx,gt_label,options);
%% Test

[pred,meanAcc,sim] = semantic_softmaxPredict(softmaxModel,x_test,y_test,labels,cla_num);
acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n',acc*100);
fprintf('meanAcc;..%0.3f%%\n',meanAcc*100);

mAP = cal_ap(cla_num,sim);
fprintf('mAP:...%0.3f%% \n', mAP*100);
% testcla_name = Test.testcla_name;
% auc(sim,cla_num,testcla_name);
% cal_PR(sim,cla_num,testcla_name);

%%
% For AwA dataset,the classification performance can be achieved 82.75% with
% attribute as semantic feature by setting lamdba = 5e-3,gamma=1e-4 and the 
% classification performance can be achieved 72.02% with wordvec as semantic
% feature by setting lambda = 1e0 and gamma = 1e-4;
% 
% For CUB dataset, the classification performance can be achieved 55.72%
% with attribute as semantic feature by setting lambda = 5e-4, gamma = 5e-6
% and the classification performance can be achieved 33.33% with wordvec as
% semantic feature by setting lambda = 1e-3, gamma = 5e-6;
% 
% For SUN dataset, the classification performance can be achieved 87.50%
% with attribute as semantic feature by setting lambda = 1e-4, gamma =
% 0.



