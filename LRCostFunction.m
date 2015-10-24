function [J,grad]=LRCostFunction(theta,X,y,lambda)
%Function to calculate Cost and Cost Gradients for a given set of 
%theta parameters

%Initialize cost to 0
J=0;

%Create array to store gradients
grad=zeros(size(theta));
m=size(X,1);

%Add bias terms to data
Xwb=[ones(m,1), X];

%Calculate cost pre-regularization 
J=(1/(2*m))*((Xwb*theta-y)'*(Xwb*theta-y));

%Add regularization term to cost
J=J+(lambda/(2*m))*(theta(2:end,1)'*theta(2:end,1));

%Calculate gradients pre-regularization
grad=(1/m)*(Xwb'*(Xwb*theta-y));

%Add regularization term to gradients
grad(2:end,1)=grad(2:end,1)+(lambda/m)*theta(2:end,1);

