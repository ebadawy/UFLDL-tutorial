function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% Couldn't get the 100 iter when subtracting max theta, always exit after 35-45 iter
% so instead I've subtracting .5 max theta and it worked fine

theta = bsxfun(@minus, theta, max(theta,[],1)/2); 
z = theta * data;
a = exp(z);
h = bsxfun(@rdivide, a, sum(a));

cost = (-1/numCases) * sum(sum(groundTruth .* log(h))) + (lambda/2) * sum(sum(theta.^2));
thetagrad = (-1/numCases) * (groundTruth - h) * data' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

