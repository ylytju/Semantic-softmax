function [mAP] = cal_ap(testnum, sim)
% ap = cal_ap(testgnd,sim)
% average precision (AP) calculation 
% Input:
% trainnum: the num of auxilary dataset for retival
% testgnd: the label of test images or attributes
% sim: the relationships between the auxilary dataset and the test images
% or attributes
test_num = length(testnum);
AP = zeros(test_num,1);
begin = 1;
for i = 1:test_num
    
    [~,pre_label] = sort(sim(:,i),'descend');
    [loc,~] = find(pre_label<=(begin+testnum(i)-1) & pre_label>=begin-1);
    ap = 0;
    for j = 1:testnum(i)
        ap = ap + j/loc(j);
    end
     AP(i) = ap/testnum(i);  
     begin = begin + testnum(i);
end
mAP = mean(AP);
