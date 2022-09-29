load hepatitisdat.sec;
data= hepatitisdat;

x= data(:,2:20);
y=data(:,1:1);
datatrain= data(1:120,:);
datatest=data(120:end,:);
data1= data(1:120,:);

xtr=x(1:120,:);
ytr=y(1:120,:);

xt=x(120:end,:);
yt= y(120:end,:);

%% Training the model 
mdlknn = fitcknn(xtr,ytr,"NumNeighbors",7,"KFold",10);
lossknn= kfoldLoss(mdlknn);
disp("initial loss" + lossknn);
%% test
[result,scores1] = predict(mdlknn,xt);
accuracyknn = sum(result == yt)/length(yt)*100;
sp = sprintf("Test Accuracy = %.2f", accuracyknn);
disp(sp);
confusionchart(y,prediict(mdlknn,y));
%% pca 
mdlknn1= fitcknn(xtr,ytr,"NumNeighbors",7,"KFold",10);
[idx,scores]=fscchi2(xtr,ytr);
bar(scores(idx));
xlabel("predictor ranks in 2nd method");
ylabel("predictor importance in 2nd method");
tokeep= idx(1:14);
selected = data(:,[tokeep,end]);
selectedtr= selected(1:120,:);
selectedtt= selected(120:end,:);
mdlsf= fitcknn(selectedtr,ytr,"KFold",10);
partloss = kfoldLoss(mdlsf);
disp("reduced loss" + partloss);
%% test feature ranking algorithm:
resultsf = predict(mdlsf,selectedtt);
accuracy2 = sum(resultsf == yt)/length(yt)*100;
sp1 = sprintf("Test Accuracy = %.2f", accuracy2);
disp(sp1);
confusionchart(y,predict(mdlknn1,y));

%% Ensemble model

ens = fitcensemble(xtr,ytr,'Method','ba', ...
   'NumLearningCycles',100,'Learners','knn','Kfold',10);
result = predict(ens,xtr);
accuracy = sum(result == ytr)/length(ytr)*100;
sp = sprintf("Test Accuracy = %.2f", accuracy);
disp(sp);
confusionchart(y,predict(ens,y));

