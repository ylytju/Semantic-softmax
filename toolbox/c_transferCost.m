function [cost,grad] = c_transferCost(theta,lambda, gamma, x_train, y_train, gt_label)
% 
% theta: the trained parameters
% lambda: weight decay parameter
% x_train: the p x N input matrix, where eacah column data (:,i)
% corresponds to a visual image
% y_train: the q x N input matrix, where each column data (:,i) corresponds
% to the semantic vector 
% 

theta = reshape(theta,size(y_train,1),size(x_train,1));
y_train = unique(y_train','rows','stable');

%% construct the affinity matrix

% y = [y_train'];
% k = 10;
% L = construct_L(y,k);


% numCases = size(x_train,2);
numClasses = size(y_train,1);
Alpha = y_train*theta;

M = bsxfun(@minus, Alpha*x_train, max(Alpha*x_train,[],1));
M = exp(M);
p = bsxfun(@rdivide,M,sum(M));
cost = -1/numClasses * gt_label(:)' * log(p(:)) + lambda/2 * sum(theta(:).^2)+gamma/2 * sum(Alpha(:).^2);
thetagrad = -1/numClasses * y_train' * (gt_label-p) * x_train' + lambda * theta + gamma * y_train' *  Alpha;

grad = [thetagrad(:)];

end
% function L = construct_L(Data,k)
% % Input Data is the training set
% % k is the k nearest neighbors
% [~,col] = size(Data);
% Dist = zeros(col,col);
% for i = 1:col
%     for j = 1:col
%        Dist(i,j) = norm(Data(:,i)-Data(:,j));
%     end
% end
% % beta = mean(mean(Dist,1),2);
% beta = 0.001;
% Correlation = zeros(col,col);
% for i = 1:col
%     tmp_i = Data(:,i);
%     for j = 1:col
%         tmp_j = Data(:,j);
%         tmp = tmp_i-tmp_j;
%         Correlation(i,j) = exp(-tmp'*tmp/(2*beta*beta)); 
%     end
% end
% % cosine_dis = pdist2(Data',Data','cosine');
% % Correlation = 1 - cosine_dis;
% 
% Similarity = zeros(col,col);
% % tmp_Dist = zeros(col,1);
% for  i = 1:col
%     tmp_Dist = Dist(:,i);
%     [~,index] = sort(tmp_Dist);
%     for j = 1:k
%         Similarity(i,index(j)) = Correlation(i,index(j));
%         Similarity(index(j),i) = Correlation(i,index(j));
%     end
% end
% 
% 
% % D = zeros(col);
% % tmp_D = sum(Similarity);
% % for i = 1:col
% %     D(i,i) = tmp_D(i);
% % end
% D = diag(sum(Similarity,2));
% L =D - Similarity;
% end
