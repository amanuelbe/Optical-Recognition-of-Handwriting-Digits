function [w2] =ldaclass_one1(all_x,all_y)
   
%% LDACLASS_ONE design a one-dimensional classifier

% In both classes there are a features which are colinear/linearly dependent, as a result
% the inverse computation couldn't be possible, since the matrix is
% singular/ Deificient Rank/.
% to avoid these problem we have two options 
% 1. Regularization
% 2. Feature selection : before computing the " within class scatter matrix "
%% Now lets proceed to the next step by using the feature selection Method

% Step 1: Filter the dataset
% Find indices of samples labeled as 1 or 8
indices_1 = find(all_y == 1);
indices_8 = find(all_y == 8);

% step 2: Filter features and labels based on selected indices
features_1 = all_x(indices_1 , :); % all filterd features for digit 1.
features_2= all_x(indices_8, :); % all filterd features for digit 8.
%% lets remove colinear feature with zero variance:
% Lets remove a zero variance feature, Becuase those feature contribute NO
% INFORMATION for the classification/discrimination task and they could
% introduce noise in our computation (Sw - singularity) and our main goal
% is to identify the image feature which are important to discriminate the
% two classes ( digit 1 and digit 8).

zero_var_features = find(var(features_1) == 0);
if ~isempty(zero_var_features)
    disp('Features with zero variance found. Removing...');
    features_1(:, zero_var_features) = [];
    disp(['Removed features: ', num2str(zero_var_features)]);
end

zero_var_features = find(var(features_2) == 0);
if ~isempty(zero_var_features)
    disp('Features with zero variance found. Removing...');
    features_2(:, zero_var_features) = [];
    disp(['Removed features: ', num2str(zero_var_features)]);
end

% calculate " between class matrix"
mu{1} = mean(features_1);
mu{2} = mean(features_2);
% calculate " within class scatter matrix"
N1 = length(indices_1); 
N2 = length(indices_8);
Sw1 = (N1-1)*cov(features_1); 
Sw2 = (N2-1)*cov(features_2);
Sw = Sw1 + Sw2 ;
% LDA vector
w2 = inv(Sw)*(mu{1}-mu{2})';  % 52x1 % Here we use inverse function
w2 = w2 ./sqrt(sum(w2.^2)); % The LDA vector w represents the optimal
%projection direction that maximizes the separation between different classes.

end
