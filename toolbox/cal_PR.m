function cal_PR(sim,cla_num,testCls)

N         = length(cla_num);
img_label = [];
for i = 1:N
   img_label = [img_label;ones(cla_num(i),1)*i];     
end
for i = 1:sum(cla_num)
   gt_label(i,img_label(i)) = 1;
end

sim = sim - ones(size(sim,1),1)*mean(sim);
for i = 1:size(sim,1)
    sim(i,:) = sim(i,:) / norm(sim(i,:));
end
for i = 1:N
    prec_rec(sim(:,i), gt_label(:,i),'holdFigure',1);
    hold on
end

legend(testCls{1},testCls{2},testCls{3},testCls{4},testCls{5},...
    testCls{6},testCls{7},testCls{8},testCls{9},testCls{10});
grid on
set(gca, 'GridLineStyle' ,'-.')
xlabel('Recall'); ylabel('Precision');

hold off

end
