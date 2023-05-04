accuracy_5x2=[];
for ndataset=1:4
    switch ndataset
        case 1, load dataset1.mat
        case 2, load dataset2.mat
        case 3, load dataset3.mat
        case 4, load dataset4.mat
        otherwise
    end
    
    accuracy_time=[];
    for ntimes=1:5
        % stratified sampling
        idx_tr=[];
        idx_te=[];
        for nclass=1:2
            u=find(labels==nclass);
            idx=randperm(numel(u));
            idx_tr=[idx_tr; u(idx(1:round(numel(idx)/2)))];
            idx_te=[idx_te; u(idx(1+round(numel(idx)/2):end))];
        end
        labels_tr=labels(idx_tr);
        labels_te=labels(idx_te);
        data_tr=data(idx_tr);
        data_te=data(idx_te);
        
%         SVM_LIN=fitcsvm(data_tr,labels_tr,'KernelFunction','linear','KernelScale',1);
%         SVM_LIN=fitcsvm(data_tr,labels_tr,'KernelFunction','gaussian','KernelScale',0.1);
%         KNN=fitcknn(data_tr,labels_tr,'Distance','Euclidean','NumNeighbors',50);
        TREE=fitctree(data_tr,labels_tr,'SplitCriterion','gdi','MaxNumSplits',20);
        
        prediction=predict(TREE,data_te);
        accuracy1 = numel(find(prediction==labels_te))/numel(labels_te)
        
%         SVM_LIN=fitcsvm(data_te,labels_te,'KernelFunction','linear','KernelScale',1);
%         SVM_LIN=fitcsvm(data_te,labels_te,'KernelFunction','gaussian','KernelScale',0.1);
%         KNN=fitcknn(data_te,labels_te,'Distance','Euclidean','NumNeighbors',50);
        TREE=fitctree(data_te,labels_te,'SplitCriterion','gdi','MaxNumSplits',20);
        
        prediction=predict(TREE,data_tr);
        accuracy2 = numel(find(prediction==labels_tr))/numel(labels_tr)
        
        accuracy=(accuracy1+accuracy2)/2
        accuracy_time(ntimes,1)=accuracy;
    end
    accuracy_time
    accuracy_5x2(ndataset,1)=mean(accuracy_time);
end
accuracy_5x2