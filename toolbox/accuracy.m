function meanAcc = accuracy( sim_mat, gt_label, testImCount)

N = numel(gt_label);            % total number
M = numel(unique(gt_label));    % class number

confusion = zeros(M, 1);           % confusion vector
top = zeros(N, 1);
for i = 1 : N
    vec = sim_mat(i,:);
    [~, indx] = sort(vec, 'descend');
    top(i) = indx(1);
end

begin = 1;
acc = zeros(M, 1);
for i = 1 : M
    confusion(i) = sum(top(begin:begin+testImCount(i)-1, 1) == gt_label(begin:begin+testImCount(i)-1, 1));   % all word
%     confusion(i) = sum(top1(begin:begin+testImCount(i)-1, 1) == i * ones(testImCount(i), 1));   % test word only
    begin = begin + testImCount(i);
    acc(i) = confusion(i) / testImCount(i);
end

meanAcc = mean(acc);