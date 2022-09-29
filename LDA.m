load hepatitisdat.sec;
data= hepatitisdat;

x= data(:,2:20);
y=data(:,1:1);
datatrain= data(1:120,:);
datatest=data(120:end,:);

xtr=x(1:120,:);
ytr=y(1:120,:);

xt=x(120:end,:);
yt= y(120:end,:);

%% model 
mdlLDA= fitcdiscr(xtr,ytr,"DiscrimType","linear","Gamma",0.1);
%% Test the model
resultLDA = predict(mdlLDA, xt);
accuracyLDA = sum(resultLDA == yt)/length(yt)*100;
sp2 = sprintf("Test Accuracy = %.2f", accuracyLDA);
disp(sp2);
confusionchart(y,predict(mdlLDA,y));

%% feature extraction:

[pcs,scrs,~,~,pctExp] = pca(data);
pareto(pctExp);
pos=find(cumsum(pctExp)>=95,1);
[idx,scores]=fscmrmr(data,y);
bar(scores(idx));
xlabel("predictor rank");
ylabel("predictor importance score");
pcaPreds= scrs(:,1:pos);
% Dividing the predictors after PCA into training and testing in the ration
% of 80:20
pcatr= pcaPreds(1:120,:);
pcatt= pcaPreds(120:end,:);
%% mdl
mdlLDA2= fitcdiscr(pcatr,ytr,"DiscrimType","linear","Gamma",0.1);
resultLDA2 = predict(mdlLDA2, pcatt);
accuracyLDA2 = sum(resultLDA2 == yt)/length(yt)*100;
print = sprintf("Test Accuracy = %.2f", accuracyLDA2);
disp(print);

loss= resubLoss(mdlLDA2);
confusionchart(y,predict(mdlLDA2,y));
%% ensembel
ens = fitcensemble(xtr,ytr,'Method','ba', ...
   'NumLearningCycles',100,'Learners','discriminant','Kfold',10);
result = predict(ens,xtr);
accuracy = sum(result == ytr)/length(ytr)*100;
sp = sprintf("Test Accuracy = %.2f", accuracy);
disp(sp);
confusionchart(y,predict(ens,y));
