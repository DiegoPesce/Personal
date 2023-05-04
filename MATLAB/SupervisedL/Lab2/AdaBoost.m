%% Testing
clearvars -except labels_te labels_tr data_tr data_te

load dataset

T=1000;
[learners, alpha] = AdaBoostCTree(data_tr, labels_tr, T);

pred = zeros([length(labels_te),T]);
for i = 1:T
    pred(:,i) = predict(learners{i}, data_te);
end

for i = 1:length(labels_te)
    [counts, classes] = groupcounts(pred(i,:)');
    boosted_pred(i) = classes( find(counts == max(counts),1) );
end
boosted_pred = boosted_pred(:);

labels_te = (labels_te-1)*2-1;

accuracy = mean( labels_te == pred, 1 );
boosted_accuracy = mean( labels_te == boosted_pred );

disp("------------- Confronto");
max_accuracy = max(accuracy)
boosted_accuracy

clearvars -except labels_te labels_tr data_tr data_te

%% AdaBoost
function [learners, alpha] = AdaBoostCTree(dataset, labels, T, methods, train_val_percentage)

    if nargin < 4
        methods = {'tree'};
    end
    if nargin < 5
        train_val_percentage = 1;
    end

    n_data = size(labels, 1);
    % start with uniform distribution
    dist = ones([n_data ,1]);
    dist = dist(:)./sum(dist); 

    minl = min(labels);
    maxl = max(labels);
    labels = 2*(labels-minl/(maxl-minl))-1; % -1 and 1

    for t = 1:T
        % train validation splitting
        % sample based on distribution
        train_idx = randsample( n_data, floor(n_data*train_val_percentage), true, dist ); 
        train_data = dataset(train_idx, :);
        train_labels = labels(train_idx);
    
        % if multiple learners are choosen, run them cyclic
        n_methods = length(methods);
        switch(lower(methods{ mod(t-1,n_methods)+1 }))
            case 'tree' 
                learners{t} = fitctree(train_data, train_labels);
            case 'svm'
                learners{t} = fitcsvm(train_data, train_labels);
            case 'lda'
                learners{t} = fitcdiscr(train_data, train_labels);
        end

        % prediction and errors on ALL the data
        pred = predict(learners{t}, dataset);
        err_idx = find(pred ~= labels);
        err = sum(dist(err_idx));
        if err > 0.5 % random generator...
            err = 0.5;
        end
        % exp error's coefficients
        alpha(t) = 0.5*log( (1-err)/err );

        % update dist
        dist = dist(:) .* exp( -alpha(t)*(labels(:).*pred(:)) );
        dist = dist(:)/sum(dist);
    end
            
end


