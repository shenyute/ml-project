function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];
a2_ = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2_];
a3 = sigmoid(a2 * Theta2');
fprintf('a3 = %d, %d\n', size(a3, 1), size(a3, 2));

for i=1:m
    y_ = zeros(num_labels, 1);
    y_(y(i), 1) = 1;
%    fprintf('y_ = %d, %d\n', size(y_, 1), size(y_, 2));
%    fprintf('*_ = %d, %d\n', size(a3(i,:), 1), size(a3(i,:), 2));
    T = - log(a3(i, :))*y_ - log(1 - a3(i, :))*(1-y_);
%    fprintf('T = %d, %d\n', size(T, 1), size(T, 2));
    J += T;
%    fprintf('J = %d, %d\n', size(J, 1), size(J, 2));
end

regular = lambda / (2*m) * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + sum(sum(Theta2(:,2:end) .* Theta2(:, 2:end))));
J = J / m + regular;

for t = 1:m
    a_1 = X(t, :);
    z2 = Theta1 * a_1';
    a2_ = sigmoid(z2);
    a2 = [1 ;a2_];
    a3 = sigmoid(Theta2 * a2);

    y_ = zeros(num_labels, 1);
    y_(y(t), 1) = 1;
    delta3 = a3 - y_;
%    fprintf('theta2 %d, %d\n', size((Theta2)));
    delta2 = (Theta2' * delta3)(2:end, :) .* sigmoidGradient(z2);
%    delta2 = delta2(2:end);
%    fprintf('delta3 %d, %d\n', size((delta3)));
%    fprintf('a2" %d, %d\n', size((a2')));
    Theta2_grad += delta3 * a2';
%    fprintf('delta2 %d, %d\n', size((delta2)));
%    fprintf('a1" %d, %d\n', size((a_1)));
    Theta1_grad += delta2 * a_1;
end
Theta2_ = Theta2;
Theta2_(:, 1) = 0;
Theta1_ = Theta1;
Theta1_(:, 1) = 0;
Theta2_grad = Theta2_grad / m + lambda / m * Theta2_;
Theta1_grad = Theta1_grad / m + lambda / m * Theta1_;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
