function auc(sim,testCount,testCls)

%
% testCount = load('testImCount.mat');
% testCount = testCount.testImCount;
% testCls = load('test_classes.mat');
% testCls = testCls.testClasses;
species = [];
testCls = testCls';
for i = 1:size(testCount)
   
    a = repmat(testCls(i),testCount(i),1);
    species = [species;a];
    clear a  
end

% sim = load('sim.mat');
% sim = sim.sim;
sim_norm = zeros(size(sim,1),size(sim,2));
p = sim_norm;
% normalization
for i = 1:size(sim,1)
   t_max = max(sim(i,:));
   t_min = min(sim(i,:));
   for j = 1:size(sim,2)
       sim_norm(i,j) = (sim(i,j)-t_min)/(t_max-t_min);   
   end
   clear t_max t_min 
end
clear i j
for i = 1:size(sim,1)
    for j = 1:size(sim,2)    
       p(i,j) = sim_norm(i,j)/sum(sim_norm(i,:));       
    end   
end
figure; clf; hold on;
X = {};
for i = 1:size(testCls,1)
    
    [X{i},Y{i},~,AUC{i}] = perfcurve(species,p(:,i),testCls{i});
    testCls{i} = strcat(testCls{i},'(','AUC=',num2str(AUC{i}),')');
    fprintf('AUC @%s: %f%% \n', testCls{i}, AUC{i}*100);
    
end
% color = {'b','g','r','c','m','y','k'};
color = {[0 0 0],[1 0 0],[0 1 0],[0 0 1],[1 1 0],[1 0 1],[0 1 1],[0.67 0 1],...
    [1 0.5 0],[0.5 0 0]};

for i = 1:size(testCls,1)
    plot(X{i},Y{i},'color',color{i},'LineWidth',2.5);
     
end
% legend(testCls{1},testCls{2},testCls{3},testCls{4},testCls{5},...
%     testCls{6},testCls{7},testCls{8},testCls{9},testCls{10});

legend(testCls{1},testCls{2},testCls{3},testCls{4},testCls{5},...
    testCls{6},testCls{7},testCls{8},testCls{9},testCls{10});
grid on
set(gca, 'GridLineStyle' ,'--')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curve')
hold off
end
