function [pred,meanAcc,sim] = semantic_softmaxPredict(c_transferModel,x_test,y_test,index,cla_num)

% c_transferModel: model trained using softmaxTrain
% x_test: the p * n matrix, where each column data (:,i) corresponds to a
% single test visual instance
% y_test: the q * N matrix, where each column data(:,i) corresponds to a
% class-level semantic vector
%
theta = c_transferModel.optTheta;

% theta = normcol_equal(theta);
Alpha =y_test' * theta;
[~,pred] = max(Alpha * x_test);

% protos = x_test'*theta';

sim = Alpha*x_test;
% sim = 1 - pdist2(y_test'*theta,x_test', 'cosine');
sim = normcol_equal(sim)';
% save('./PR_fea/CUB_att_sim.mat','sim');
% vis_prototype = W'*Test_y;
meanAcc = accuracy(sim,index,cla_num);
end