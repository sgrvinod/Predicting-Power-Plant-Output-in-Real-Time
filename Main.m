clear; close all; clc

%Load data to be trained
load('AllData.csv')
X=AllData(:,1:4);
y=AllData(:,5);

%Add polynomial features
%Add quadratic terms
b=size(X,2)+1;
for i=1:4,
    for j=i:4,
        X(:,b)=X(:,i).*X(:,j);
        b=b+1;
    end;
end;
%Add cubic terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            X(:,b)=X(:,i).*X(:,j).*X(:,k);
            b=b+1;
        end;
    end;
end;
%Add 4-degree terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            for l=k:4,
                X(:,b)=X(:,i).*X(:,j).*X(:,k).*X(:,l);
                b=b+1;
            end;
        end;
    end;
end;
%Add 5-degree terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            for l=k:4,
                for m=l:4,
                    X(:,b)=X(:,i).*X(:,j).*X(:,k).*X(:,l).*X(:,m);
                    b=b+1;
                end;
            end;
        end;
    end;
end;
%Add 6-degree terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            for l=k:4,
                for m=l:4,
                    for n=m:4,
                        X(:,b)=X(:,i).*X(:,j).*X(:,k).*X(:,l).*X(:,m).*X(:,n);
                        b=b+1;
                    end;
                end;
            end;
        end;
    end;
end;
%Add 7-degree terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            for l=k:4,
                for m=l:4,
                    for n=m:4,
                        for o=n:4,
                            X(:,b)=X(:,i).*X(:,j).*X(:,k).*X(:,l).*X(:,m).*X(:,n).*X(:,o);
                            b=b+1;
                        end;
                    end;
                end;
            end;
        end;
    end;
end;
%Add 8-degree terms
for i=1:4,
    for j=i:4,
        for k=j:4,
            for l=k:4,
                for m=l:4,
                    for n=m:4,
                        for o=n:4,
                            for q=o:4,
                                X(:,b)=X(:,i).*X(:,j).*X(:,k).*X(:,l).*X(:,m).*X(:,n).*X(:,o).*X(:,q);
                                b=b+1;
                            end;
                        end;
                    end;
                end;
            end;
        end;
    end;
end;
%Feature scale between 0 and 1
minx=min(X,[],1);
maxx=max(X,[],1);
minxm=repmat(minx,size(X,1),1);
maxxm=repmat(maxx,size(X,1),1);
temp=(X-minxm)./(maxxm-minxm);
X=temp;
          
%Split into training and validation sets
train.indices=randperm(size(X, 1),ceil(0.7*size(X, 1)));
cv.indices=setdiff(1:size(X, 1),train.indices);
Xtrain=X(train.indices,:);
ytrain=y(train.indices,:);
Xcv=X(cv.indices,:);
ycv=y(cv.indices,:);

%Initialize thetas
inittheta=zeros(size(X,2)+1,1);

%Initialise lambda
lambda=0;

%Create function to return cost and gradients
initJ=0;
[initJ,gradvec]=LRCostFunction(inittheta,Xtrain,ytrain,lambda);

%Set options for fminunc
options = optimset('GradObj', 'on','MaxIter',10000);

%Run fminunc to obtain the optimal theta
%This function will return theta and the cost 
[theta, cost, exitflag] = ...
	fminunc(@(t)(LRCostFunction(t, Xtrain, ytrain,lambda)), inittheta, options);

%Predict
pcv=predict(theta,Xcv);
ptrain=predict(theta,Xtrain);

%Find training and cross validation standard errors
mcv=size(Xcv,1);
mtrain=size(Xtrain,1);
stderrcv=(1/mcv)*sum(abs(pcv-ycv))
stderrtrain=(1/mtrain)*sum(abs(ptrain-ytrain))

