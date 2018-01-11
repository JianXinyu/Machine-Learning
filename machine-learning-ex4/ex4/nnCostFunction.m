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

%PART 1:Feedforword the nerual network and return the cost in the variable J
%--------------------------------------------------------------
% a1 = [ones(m,1) X];
% z2 = a1 * Theta1';
% a2 = [ones(m,1) sigmoid(z2)];
% z3 = a2 * Theta2';
% a3 = sigmoid(z3);
% h = a3;
% 
% %Compute J
% for k = 1:num_labels
%     yk = (y == k);
%     hk = h(:,k);
%     Jk = -1/m * sum(yk .* log(hk) + (1 - yk) .* log(1 - hk));
%     J = J + Jk;
% end
% 
% %Regularization
% 
% %%Remove Bias
% Theta1NoBias = Theta1(:,2:end);
% Theta2NoBias = Theta2(:,2:end);
% 
% reg = lambda / (2 * m) * (sum(sum(Theta1NoBias .^ 2)) + sum(sum(Theta2NoBias .^2)));
% J = J + reg;

%---------------------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
% Delt1 = 0;
% Delt2 = 0;
% for i = 1:m
%     %1.Feedforward
%     a1 = [1; X(i, :)'];
%     z2 = Theta1 * a1;
%     a2 = [1; sigmoid(z2)];
%     z3 = Theta2 * a2;
%     a3 = sigmoid(z3);
%     %2.For each output unit k in layer 3 (the output layer), set
%     delt3 = a3 - y(i,:)';
%     %3.For the hidden layer l = 2, set
%     delt2 = (Theta2NoBias' * delt3) .* sigmoidGradient(z2);
%     %4.Accumulate the gradient
%     Delt2 = Delt2 + delt3 * a2';
%     Delt1 = Delt1 + delt2 * a1';
% end
% %5.Obtain the (unregularized) gradient
% Theta2_grad = Delt2 / m;
% Theta1_grad = Delt1 / m;

%recode y
recoded_y = zeros(m,max(y));
for i = 1:m
    recoded_y(i,y(i,1)) = 1;
end   

%forward pass 
temp = [ones(m,1) X];
z2 = (temp*Theta1');
a2 = sigmoid(z2);
temp2 = [ones(m,1) a2];
z3 = (temp2*Theta2');
a3 = sigmoid(z3);
[~,h] = max(a3,[],2);

%compute cost
for i = 1:m
    costOfInstance = (sum((recoded_y(i,:).*log(a3(i,:))+(1-recoded_y(i,:)).*log(1-a3(i,:)))));
    J = J+costOfInstance;
end

J = J*(-1/m);

%regularization
output_thetas = 0;
hidden_thetas = 0;
reg=lambda/(2*m);

temp_theta2 = Theta2(:,2:size(Theta2,2));
for i = 1:size(Theta2,1)
    output_thetas = output_thetas+sum(temp_theta2(i,:).^2); 
end

temp_theta1 = Theta1(:,2:size(Theta1,2));
for i = 1:size(Theta1,1)
    hidden_thetas = hidden_thetas+sum(temp_theta1(i,:).^2); 
end
J = J+reg*(output_thetas+hidden_thetas);

%backpropagation-- Assumumption about 3 layer structure

%for each example
for t = 1:m
    %forward pass

    a1 = X(t,:);
    a1 = [1 a1];
    z2 = a1*Theta1';
    a2 = sigmoid(z2);
    a2 = [1 a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    
    %error at output layer
    delta3 = (a3-recoded_y(t,:));
    
    %error at hidden layer
    delta2 = (delta3*Theta2).*[1 sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    
    %gradient calculation
    Theta2_grad = (Theta2_grad+(delta3'*a2));
    Theta1_grad = (Theta1_grad+(delta2'*a1));
end

Theta1_grad = (1/m).*Theta1_grad;
Theta2_grad = (1/m).*Theta2_grad;

%Regularize Gredient

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);
% -------------------------------------------------------------




% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
