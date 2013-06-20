function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


h = sigmoid(X*theta);
J = sum(y'*log(h) + (1-y)'*log(1-h)) * -1/m;
grad = ((h-y)'*X)/m;

% make theta(0) to be zero, so regression is not calculated for it.
theta_partial =  theta;
theta_partial(1) = 0;

J = J + ((lambda/(2*m)) * (theta_partial'*theta_partial));
grad = grad + (lambda/m * theta_partial');

grad = grad(:);



% =============================================================

end
