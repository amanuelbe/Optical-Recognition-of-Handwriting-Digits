
function [w1] =ldaclass_one(all_x,all_y)

%% LDACLASS_ONE design a one-dimensional classifier

% In both classes there are a features which are colinear/linearly dependent, as a result
% the inverse computation couldn't be possible, since the matrix is
% singular/ Deificient Rank/.
% to avoid these problem we have two options
% 1. Regularization
% 2. Feature selection : before computing the " within class scatter matrix "

%% Now lets proceed to the next step by using the Regularization Method

% Step 1: Filter the dataset
% Find indices of samples labeled as 1 or 8
indices_1 = find(all_y == 1);
indices_8 = find(all_y == 8);

% step 2: Filter features and labels based on selected indices
features_1 = all_x(indices_1 , :); % all filterd features for digit 1.
features_2= all_x(indices_8, :); % all filterd features for digit 8.

% Calculate "between class matrix"
mu{1} = mean(features_1); % digit 1
mu{2} = mean(features_2); % digit 8
% calculate "within class scatter matrix"
N1 = length(indices_1);
N2 = length(indices_8);
Sw1 = (N1-1)*cov(features_1);
Sw2 = (N2-1)*cov(features_2);

% Regularization parameter (small constant)
lambda = 0.01; % Adjust the value as needed

% Add regularization term to covariance matrices
Sw1_reg = Sw1 + lambda * eye(size(Sw1)); % creates an identity matrix with the same size as Sw1,
% and lambda * eye(size(Sw1)) scales this identity matrix by the regularization parameter
Sw2_reg = Sw2 + lambda * eye(size(Sw2)); % same like Sw1.

% Compute the combined regularized covariance matrix
Sw_reg = Sw1_reg + Sw2_reg;

% LDA vector using regularized pseudo-inverse
w1 = pinv(Sw_reg) * (mu{1} - mu{2})';  % dx1 % and we use pinv function to do the inverse of the regulirized matrix.

% Normalize the LDA vector
w1 = w1 ./ sqrt(sum(w1.^2)); % To
end
