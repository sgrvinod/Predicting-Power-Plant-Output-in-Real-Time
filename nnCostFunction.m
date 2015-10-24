function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


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

%forwardpropagation
a1=X;
a1wb=[ones(size(X, 1),1),X];
z2=a1wb*Theta1'; 
a2=tanh(z2);
a2wb=[ones(size(a2, 1),1),a2];
z3=a2wb*Theta2';
a3=z3;
htheta=a3;

%cost function
J=(1/(2*m))*sum(sum((htheta-y).^2));
J=J+(lambda/(2*m))*(sum(sum(Theta1(:,2:size(Theta1, 2)).^2))+sum(sum(Theta2(:,2:size(Theta2, 2)).^2)));


%backpropagation
capdelta1=zeros(size(Theta1));
capdelta2=zeros(size(Theta2));
grad1=zeros(size(Theta1));
grad2=zeros(size(Theta2));

for i=1:m,
  a1s=(X(i,:))';
  a1wbs=[1;a1s];
  z2s=Theta1*a1wbs;
  a2s=tanh(z2s);
  a2wbs=[1;a2s];
  z3s=Theta2*a2wbs;
  a3s=z3s;
  hthetas=a3s;

  delta3=hthetas-y(i);
  delta2wb=(Theta2'*delta3).*(1-((tanh([1;z2s])).^2));
  delta2=delta2wb(2:end);

  capdelta1=capdelta1+delta2*a1wbs';
  capdelta2=capdelta2+delta3*a2wbs';
end;

grad1=(1/m)*capdelta1;
grad2=(1/m)*capdelta2;
grad1(:,2:size(grad1, 2))=grad1(:,2:size(grad1, 2))+(lambda/m)*Theta1(:,2:size(Theta1, 2));
grad2(:,2:size(grad2, 2))=grad2(:,2:size(grad2, 2))+(lambda/m)*Theta2(:,2:size(Theta2, 2));

grad=[grad1(:);grad2(:)];
end













