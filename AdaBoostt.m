function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
 
 
N = length(y_tr);
M = n_trees;
D = zeros(N,M+1);
D(:,1) = 1/N;
h = zeros(N,M);
H = zeros(N,M);
H_final = zeros(N,M);
epsilon = zeros(1,M);
alpha = zeros(1,M);
Z = zeros(1,M);
%y_hat = zeros(N,1);
errors = zeros(N,M);
num_errors = zeros(N,1);
perc_misclassified = zeros(N,1);
 
 
% Map y to {-1,1}
%minY = min(y_tr);
%y_tr = y_tr - (minY + 1);
%y_te = y_te - (minY + 1);
 
for t = 1:M
    stump = fitctree(X_tr,y_tr,'MaxNumSplits',1,'SplitCriterion','deviance','Weights',D(:,t));
    h(:,t) = predict(stump,X_tr);
    %h(:,t) = predict_weak_learner(stump,X_tr);
    logical_classification = h(:,t)==y_tr;
    correctly_classified = find(logical_classification == 1);
    misclassified = find(logical_classification == 0);
    num_misclassified = length(misclassified);
    %epsilon(t) = num_misclassified/N;
    epsilon(t) = sum(D(misclassified,t));
    alpha(t) = 1/2 * log((1-epsilon(t))/epsilon(t));
    alpha(t);
    Z(t) = 2*sqrt(epsilon(t)*(1-epsilon(t)));
    D(correctly_classified,t+1) = D(correctly_classified,t)*exp(-alpha(t)) / Z(t);
    D(misclassified,t+1) = D(misclassified,t)*exp(alpha(t)) / Z(t);
    %D(correctly_classified,t+1) = D(correctly_classified,t)*exp(-alpha(t)) / sum(D(:,t));
    %D(misclassified,t+1) = D(misclassified,t)*exp(alpha(t)) / sum(D(:,t));
    H(:,t) = alpha(t)*h(:,t);
end
 
 
H_final(:,1) = H(:,1);
errors(:,1) = (sign(H_final(:,1)) ~= y_tr);
for t = 2:M
    H_final(:,t) = H_final(:,t-1) + H(:,t);
    errors(:,t) = (sign(H_final(:,t)) ~= y_tr);
end
num_errors = sum(errors,1);
perc_misclassified = num_errors/N;
train_err = perc_misclassified(M);
test_err = train_err;
 
 
figure
plot(1:M, perc_misclassified,'o')
ylabel('Percent Misclassified')
xlabel('Number of Boosting Rounds')
 
% y_hat = sign(sum(H,2));
% training_errors = (y_hat ~= y_tr);
% num_training_errors = sum(training_errors)
% train_err = num_training_errors/length(y_tr);
% test_err = train_err;
 
 
end