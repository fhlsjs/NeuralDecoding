clear;
load midterm_train;
%Estimate parameters
M1=3100;
x=kin';
y=rate';
Ah=x(:,2:end)*x(:,1:end-1)'*inv(x(:,1:end-1)*x(:,1:end-1)');
Wh=(x(:,2:end)*x(:,2:end)'-Ah*x(:,1:end-1)*x(:,2:end)')/(M1-1);
y(1:3,1)
%Set initial values to be [1;1;1;1;1]



a(:,:,1)=[2 1; 3 2];
a(:,:,2)=[3 6; 4 9];
b=[1 2];
a(:,:,1)+a(:,:,2)
x=[ones(1,M1); x];

a(:,:,1)=zeros(5,5);
for k=2:(M1+1)
    a(:,:,k)=a(:,:,k-1)-exp(est(:,i,j)'*x(:,k-1))*x(:,k-1)*x(:,k-1)';
    

N=1000;
est(:,:,1)=ones(5,42);
for i=1:42
    for j=2:N
        a(:,:,1)=zeros(5,5);
        b(:,1)=zeros(5,1);
        for k=2:(M1+1)
            a(:,:,k)=a(:,:,k-1)-exp(est(:,i,j-1)'*x(:,k-1))*x(:,k-1)*x(:,k-1)';
            b(:,k)=b(:,k-1)+y(i,k-1)*x(:,k-1)-exp(est(:,i,j-1)'*x(:,k-1))*x(:,k-1);
        end
    est(:,i,j)=est(:,i,j-1)-inv(a(:,:,k))*b(:,k);
    end
end

        
        
% Inhomogeneous Poisson process
clear; 
%Work on training data to estimate parameters
load midterm_train; 
[M, C] = size(rate);  %Row and column dimensions
d = size(kin,2);
%Centralize the data
kin_mean = mean(kin);
kin = kin - ones(M,1)*kin_mean;
% Fit the kinematic model  v_{k+1} = A*v_k+w_k
% Ah, Wh are the same as in a Kalman filter model
a = kin(2:M,:)'; 
b = kin(1:M-1,:)';
Ah = a*b'*inv(b*b');   
Wh = (a-Ah*b)*(a-Ah*b)'/(M-1);

% Estimate mu_c and alpha_c using MLE method
for c = 1:C
    alpha_old = zeros(d+1,1);
    alpha_new = alpha_old + 1;
    % Newton-Raphson method, closed-form formulae in lecture notes
    count(c) = 0;
    while norm(alpha_new-alpha_old) > 1e-2
        count(c) = count(c) + 1;  
        alpha_old = alpha_new;
        Kin = [ones(M,1) kin];
        nmt = Kin'*rate(:,c) - Kin'*exp(Kin*alpha_old);
        dnt = - Kin'.*(ones(d+1,1)*exp(Kin*alpha_old)')*Kin;
        alpha_new = alpha_old - dnt\nmt;
    end
    alpha(:,c) = alpha_new;
end

%Decoding using Point Process filter on the test data
load midterm_test;
M2 = size(rate, 1);
%Set initial kinematic (x) values
x_m(:,1) = zeros(d,1);  
x(:,1) = x_m(:,1);
P_m(:,:,1) = zeros(d); 
P(:,:,1) = zeros(d);

%Reconstruction by PPF
for k = 2:M2
    P_m(:,:,k) = Ah*P(:,:,k-1)*Ah'+Wh;
    x_m(:,k) =  Ah*x(:,k-1);
    P(:,:,k) = inv(inv(P_m(:,:,k)) + ...
                alpha(2:end,:)*diag(exp([1 x_m(:,k)']*alpha))*alpha(2:end,:)');
    x(:,k) = x_m(:,k)+P(:,:,k)*(alpha(2:end,:)*(rate(k,:)-exp([1 x_m(:,k)']*alpha))');
end
%Centralization "recovery"
x = x + kin_mean'*ones(1,M2);
% R^2 Error 
r2 = 1 - sum((x'-kin).^2)./ ...
                 sum((kin-ones(M2,1)*kin_mean).^2);  
disp(['R2 = ' num2str(r2)]);
%ylab=['x-position', 'y-position', 'x-velocity', 'y-velocity'];
%tit1=['Reconstruction of PPF Algorithm, x-position', ...
      %'Reconstruction of PPF Algorithm, y-position', ...
      %'Reconstruction of PPF Algorithm, x-velocity', ...
      %'Reconstruction of PPF Algorithm, y-velocity'];
figure(1);
%Here d=4, so it's good to use a loop
for i = 1:d
    subplot(2,2,i);
    plot(1:M2,kin(:,i)','r--','linewidth', 2);
    hold on;
    plot(1:M2, x(i,:),'b--','linewidth', 2);
    legend('True','Reconstructed','Location','best');
    s=subplot(2,2,i);
    title(s,'Reconstruction of PPF Algorithm', 'Fontsize', 12);
end

%Reconstruction by SMC
load midterm_test;
M2 = size(rate, 1);
K = 500;  %sample size, subject to change

%Assign initial values to the sample data set
s = zeros(d,K,M2);  
weight = ones(K,M2)/K;  
Weight = cumsum(weight); 
x_h = zeros(d,M2);

for t = 2:M2
   if mod(t,100) == 0
       disp(sprintf('t = %d', t));
   end
   r = rand(1,K);
   j = sum(Weight(:,t-1)*ones(1,K) < ones(K,1)*r)+1;
   s_p = s(:,j,t-1);
   s(:,:,t) = Ah*s_p + mvnrnd(zeros(1,d),Wh,K)';
   %Compute the (log) likelihood
   for k = 1:K
       lambda = exp(alpha'*[1; s(:,k,t)]); 
       log_like(k,1) = sum(log(poisspdf(rate(t,:)', lambda)));
   end      
   % the weights at step t
   weight(:,t) = ones(K,1)./sum(exp(ones(K,1)*log_like'-log_like*ones(1,K)),2);
   Weight(:,t) = cumsum(weight(:,t));
   % estimate the kinematic (x) values
   x_h(:,t) = s(:,:,t)*weight(:,t);   
end
%Centralization "recovery"
x_h = x_h + kin_mean'*ones(1,M2);
%R^2 Error 
r2 = 1 - sum((x_h'-kin).^2)./ ...
                 sum((kin-ones(M2,1)*kin_mean).^2);  
disp(['R2 = ' num2str(r2)]);
%Visualization display
figure(2);
for i = 1:d
    subplot(2,2,i);
    plot(1:M2,kin(:,i)','r--','linewidth', 2);
    hold on;
    plot(1:M2, x_h(i,:),'b--','linewidth', 2);
    legend('True','Reconstructed','Location','best');
    s=subplot(2,2,i);
    title(s,'Reconstruction of SMCM, K=500', 'Fontsize', 12);
end
