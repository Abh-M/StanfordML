function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
j_theta = 0.0;
for i = 1:m
  h_theta = sigmoid(X(i,:)*theta);
  j_theta = j_theta + (-y(i,:))*log(h_theta) - (1-y(i,:))*log(1-h_theta);
endfor
J = j_theta/m;

for j = 1:size(theta)
  pd = 0.0
  for i = 1:m
    h_theta = sigmoid(X(i,:)*theta);
    pd = pd + (h_theta-y(i,:))*X(i,j);
  endfor  
  pd = pd/m;
  grad(j) = pd;
endfor







% =============================================================

end
