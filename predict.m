function p=predict(theta, X)
%Function to calculate hypotheses

%Create array to store prediction
p=zeros(size(X, 1), 1);

%Add bias term to data
Xwb=[ones(size(X,1),1), X];

%Calculate linear regression hypotheses
p=Xwb*theta;

end
