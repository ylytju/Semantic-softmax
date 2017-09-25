function [meanAcc, vis_prototype] = predict_sae(W,Test)

Test_x = Test.im_fea;
Test_y = normcol_equal(Test.semantic_uni);
cla_num = Test.cla_num;
index = Test.index;

sim = Test_x'*W'*Test_y;

% sim = 1 - pdist2(Test_x'*W',Test_y', 'cosine');
% sim = 1 - pdist2(Test_x',Test_y'*W, 'cosine');
vis_prototype = W'*Test_y;
meanAcc = accuracy(sim,index,cla_num);
end