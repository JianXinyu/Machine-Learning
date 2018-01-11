function [C, sigmasam] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigmasam = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% sample = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Csam = sample;
% sigmasam = sample;
% SizeSample = size(sample, 2);
% predictionerror = zeros(SizeSample, SizeSample);
% 
% for row=1:SizeSample
%     for col=1:SizeSample
%         model = svmTrain(X, y, Csam(row), @(x1, x2) gaussianKernel(x1, x2, sigmasam(col))); 
%         predictions = svmPredict(model, Xval);
%         predictionerror(row, col) = mean(double(predictions ~=yval));
%     end
% end
% val = min(min(predictionerror));
% [row col] = find(val == predictionerror);
% C = Csam(row);
% sigma = sigmasam(col);

bestPrediction = 10;
test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for i = 1:length(test)
  for j= 1:length(test)
      C_t = test(i);
      sigma_t = test(j);
      model= svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t));
      predictions = svmPredict(model, Xval);
      prediction = mean(double(predictions ~= yval));
      if prediction < bestPrediction;
          bestPrediction = prediction;
          C = C_t;
          sigmasam = sigma_t;
      end
  end
end

% =========================================================================

end
