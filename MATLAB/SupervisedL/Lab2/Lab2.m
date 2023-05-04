%% load dataset
clear
load dataset.mat

%% stratified sampling
rng('default') % For reproducibility
idx_f1=[];
idx_f2=[];
for nclass=1:2
    u=find(labels_tr==nclass);
    idx=randperm(numel(u));
    idx_f1=[idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2=[idx_f2; u(idx(1+round(numel(idx)/2):end))];
end
labels_f1=labels_tr(idx_f1);
labels_f2=labels_tr(idx_f2);
data_f1=data_tr(idx_f1,:);
data_f2=data_tr(idx_f2,:);

%% train level-1 classifiers on fold1
mdls={};
% SVM with gaussian kernel
rng('default') % For reproducibility
mdls{1}= fitcsvm(data_f1, labels_f1, 'KernelFunction','gaussian','KernelScale',5);

% SVM with polynomial kernel
rng('default') % For reproducibility
mdls{2}= fitcsvm(data_f1, labels_f1, 'KernelFunction','polynomial','KernelScale',10);

% decision tree
rng('default') % For reproducibility
mdls{3} = fitctree(data_f1, labels_f1, 'SplitCriterion','gdi','MaxNumSplits',20);

% Naive Bayes
rng('default') % For reproducibility
mdls{4} = fitcnb(data_f1, labels_f1);

% Ensemble of decision trees
rng('default') % For reproducibility
mdls{5} = fitcensemble(data_f1, labels_f1);

%% predictions on fold2 (to be used to train the meta-learner)

N=numel(mdls);
Predictions=zeros(size(data_f2,1),N);
Scores=zeros(size(data_f2,1),N);
for ii=1:N
    [predictions, scores] = predict(mdls{ii},data_f2);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
end

% TBD: compare the performance of the meta-classifier when trained
% on Predictions (i.e. predicted classes) instead of Scores

% TBD: compare the performance of the meta-classifier when the training sl
% split is not performed and the same data is used to train the
% level-1 classifiers and the meta-classifier

%% train the stacked classifier on fold2
rng('default') % For reproducibility
% stckdMdl = fitcensemble(Scores, labels_f2);
stckdMdl = fitcensemble(Scores, labels_f2, 'Method','AdaBoostM1');%,'NumLearningCycles', 100);
stckdMdl=fitcensemble(Scores,labels_f2,'Method','Bag');
mdls{N+1} = stckdMdl;

%% measure performance on the test set
ACC=[];
rng('default');
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);
for ii=1:N
    [predictions, scores] = predict(mdls{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC(ii)= numel(find(predictions==labels_te))/numel(labels_te);
end
predictions = predict(mdls{N+1}, Scores);
ACC(N+1)= numel(find(predictions==labels_te))/numel(labels_te);
ACC


