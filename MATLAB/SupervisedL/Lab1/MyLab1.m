clear
% algorithm params
load dataset1.mat
[N,M] = size(data);
train_perc = 0.5;

% 5x2 cross validation

for iteration = 1:5
    
    % split data
    train_indexes_1 = randperm(N, train_perc*N);
    train_indexes_2 = setdiff(1:N,train_indexes_1); %setdiff returns ordered array
    train_indexes_2 = train_indexes_2( randperm(length(train_indexes_2)) ); %shuffle
    
    %train validation split
    labels_1 = labels(train_indexes_1); %labels using train1
    data_1 = data(train_indexes_1 ,:); %data using train1
    labels_2 = labels(train_indexes_2); %labels using train2
    data_2 = data(train_indexes_2,:); %data using train2
    
    %%%%%%%% linear discriminant analysis
    %first round for cv
    LDA_1 = fitcdiscr(data_1,labels_1);
    pred_lda_on_data_2 = predict(LDA_1, data_2);
    lda_err_on_data_2 = mean( (pred_lda_on_data_2 - labels_2).^2 ); %mse
    %second round for sv
    LDA_2 = fitcdiscr(data_2,labels_2);
    pred_lda_on_data_1 = predict(LDA_2, data_1);
    lda_err_on_data_1 = mean( (pred_lda_on_data_1 - labels_1).^2 ); %mse
    %accuracy...

    %%%%%%%%
    
    %%%%%%%% tree
    %first round
    tree_1 = fitctree(data_1, labels_1);
    pred_tree_on_data_2 = predict(tree_1, data_2);
    tree_err_on_data_2 = mean( (pred_tree_on_data_2 - labels_2).^2 ); %mse
    %second round
    tree_2 = fitctree(data_2, labels_2);
    pred_tree_on_data_1 = predict(tree_2, data_1);
    tree_err_on_data_1 = mean( (pred_tree_on_data_1 - labels_1).^2 ); %mse
    %accuracy...

    %%%%%%%%%
    
    
    %%%%%%%%% support vector machine
    %first round
    svm_1 = fitcsvm(data_1,labels_1,'KernelFunction','rbf',...
        'Standardize',true);
    pred_svm_on_data_2 = predict(svm_1, data_2);
    svm_err_on_data_2 = mean( (pred_tree_on_data_2 - labels_2).^2 ); %mse
    %second round
    svm_2 = fitcsvm(data_2,labels_2,'KernelFunction','rbf',...
        'Standardize',true);
    pred_svm_on_data_1 = predict(svm_2, data_1);
    svm_err_on_data_1 = mean( (pred_tree_on_data_1 - labels_1).^2 ); %mse
    %accuracy...

    %%%%%%%%%
end
clearvars 
