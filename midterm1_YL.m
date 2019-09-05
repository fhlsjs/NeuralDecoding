clear;
load midterm_train;
%Estimate parameters
x=kin';
y=rate';
M1=3100;
M2=910;
Ah=x(:,2:end)*x(:,1:end-1)'*inv(x(:,1:end-1)*x(:,1:end-1)');
Wh=(x(:,2:end)*x(:,2:end)'-Ah*x(:,1:end-1)*x(:,2:end)')/(M1-1);
Hh=y(:,1:end)*x(:,1:end)'*inv(x(:,1:end)*x(:,1:end)');
Qh=1/M1*(y(:,1:end)*y(:,1:end)'-Hh*x(:,1:end)*y(:,1:end)');
Wh=ceil(100 * Wh)/100; %appoximate Wh to 0.001


load midterm_test;
x2=kin';
y2=rate';
xh(:,1)=[0; 0; 0; 0];
P(:,:,1)=zeros(4,4);
xhminus(:,1)=[0; 0; 0; 0];
Pminus(:,:,1)=zeros(4,4);
K(:,:,1)=zeros(4,42);
for k=2:(M2+1)
    xhminus(:,k)=Ah*xh(:,k-1);
    Pminus(:,:,k)=Ah*P(:,:,k-1)*Ah'+Wh;
    K(:,:,k)=Pminus(:,:,k)*Hh'*inv(Hh*Pminus(:,:,k)*Hh'+Qh);
    xh(:,k)=xhminus(:,k)+K(:,:,k)*(y2(:,k-1)-Hh*xhminus(:,k));
    P(:,:,k)=(eye(4)-K(:,:,k)*Hh)*Pminus(:,:,k);
end

%Compare true and estimated states (for the first two components)
figure(1);
subplot(2,1,1);
plot(1:M2,x2(1,:),'b-');
hold on;
plot(1:M2,xh(1,2:911),'r-');
title('Component 1 (x-position)');
subplot(2,1,2);
plot(1:M2,x2(2,:),'b-');
hold on;
plot(1:M2,xh(2,2:911),'r-');
title('Component 2 (y-position)');

%Compute R^2 (component-wise) for filtering estimation
RSquaref(1) = 1-sum((x2(1,:)-xh(1,2:911)).^2)/(909*var(x2(1,:)));
RSquaref(2) = 1-sum((x2(2,:)-xh(2,2:911)).^2)/(909*var(x2(2,:)));
n=[20 50 100 500];
r=mvnrnd(zeros(1,4),eye(4),n(1))';
xseqmc(:,1)=zeros(4,1);
xtilt(:,:,1)=mvnrnd((Ah*r)',Wh);
wei(:,1)=zeros(n(1),1);
weinor(:,1)=zeros(n(1),1);
for i=2:(M2+1)
    xtilt(:,:,i)=mvnrnd((Ah*xtilt(:,:,i-1)')',Wh);
    wei(:,i)=mvnpdf(repmat(y2(:,i)',n(1),1),(Hh*xtilt(:,:,i)')',Qh);
    weinor(:,i)=wei(:,i)/sum(wei(:,i));
    xseqmc(:,i)=(weinor(:,i)'*xtilt(:,:,i))';
end


%%%Kalman filter model%%%
clear; 
%Work on training data to estimate parameters
load midterm_train; 
[M, C] = size(rate); %Row and column dimensions
%Centralize the data
mean_pos = mean(kin(:,1:2));  %Only need the positions (first 2 components of x) 
%ones(M,1) is M by 1 and mean_pos is 1 by 2
kin(:,1:2) = kin(:,1:2) - ones(M,1)*mean_pos;
mean_rate = mean(rate);
%# of rows of rate = # of rows of kin
rate = rate - ones(M,1)*mean_rate;

% Model identification
a = kin(2:M,:)'; 
b = kin(1:M-1,:)';
c = rate';  
d = kin';
%The definition of a, b, c and d smartly manipulates matrix multiplication
%in the closed-form formulae
%The estimates are denoted as Ah, Wh, Hh, and Qh
Ah = a*b'*inv(b*b');   
Wh = (a-Ah*b)*(a-Ah*b)'/(M-1);
Hh = c*d'*inv(d*d');   
Qh = (c-Hh*d)*(c-Hh*d)'/M;


%Neural decoding on test data by Kalman filter algorithm
%Loading test data has to be done here and we need to redefine dimensions
load midterm_test;
[M, C] = size(rate);
%C actually equals 42
%Here we use the mean values of the training data 
%since we "pretend" not to know kin in the test data
rate = rate - ones(M,1)*mean_rate;
d = size(kin,2);
%Define initial values of all estimates
x_minus = zeros(d, M);
x = zeros(d, M);
P_minus = zeros(d, d, M);
P = zeros(d, d, M);
K = zeros(d, C, M);

for k = 2:M
    % prior estimation
    P_minus(:,:,k) = Ah*P(:,:,k-1)*Ah'+Wh;
    x_minus(:,k) = Ah*x(:,k-1);
    % posterior estimation
    K(:,:,k) = P_minus(:,:,k)*Hh'*inv(Hh*P_minus(:,:,k)*Hh'+Qh);
    P(:,:,k) = (eye(d)-K(:,:,k)*Hh)*P_minus(:,:,k);
    x(:,k) = x_minus(:,k)+K(:,:,k)*(rate(k,:)'-Hh*x_minus(:,k));
end

%Centralization "recovery"
x(1:2,:) = x(1:2,:) + mean_pos'*ones(1,M);
x_minus(1:2,:) = x_minus(1:2,:) + mean_pos'*ones(1,M);  

%Show the decoding of x and y positions
figure(1);
subplot(2,1,1);
plot(1:M, kin(:,1),'r--','linewidth', 2);
hold on;
plot(1:M, x(1,:)','b--','linewidth',2); 
ylabel('x-position','FontSize', 10);
legend('True','Reconstructed','Location','best');
s1=subplot(2,1,1);
title(s1,'Reconstruction of KF Algorithm, x-position', 'Fontsize', 12);
subplot(2,1,2);
plot(1:M, kin(:,2),'r--','linewidth', 2);
hold on;
plot(1:M, x(2,:)','b--','linewidth',2);
ylabel('y-position','FontSize', 10);
legend('True','Reconstructed','Location','best');
s2=subplot(2,1,2);
title(s2,'Reconstruction of KF Algorithm, y-position', 'Fontsize', 12);
xlabel('time (t)');

% R^2 Error 
r2 = 1 - sum((x(1:2,:)'-kin(:,1:2)).^2)./ ...
                 sum((kin(:,1:2)-ones(M,1)*mean_pos).^2);  
disp(['R2 = ' num2str(r2)]);

% Sequential Monte Carlo algorithm
K = 20;   % sample size, subject to change
% initialize the sample data set
s = zeros(d,K,M);  
weight = ones(K,M)/K;  
Weight = cumsum(weight); %Accumulative weights
x_h = zeros(d,M);

for t = 2:M
   if mod(t,100) == 0
       disp(sprintf('t = %d', t));
   end
   r = rand(1,K);
   j = sum(Weight(:,t-1)*ones(1,K) < ones(K,1)*r)+1;
   s_p = s(:,j,t-1);
   s(:,:,t) = Ah*s_p + mvnrnd(zeros(1,d),Wh,K)';

   %compute the (log) likelihood
   for k = 1:K
         log_like(k,1) = -0.5*(log(det(Qh))+(rate(t,:)'-Hh*s(:,k,t))'...
            *inv(Qh)*(rate(t,:)'-Hh*s(:,k,t)));
   end      
   % the weights at step t
   weight(:,t) = ones(K,1)./sum(exp(ones(K,1)*log_like'-log_like*ones(1,K)),2);
   Weight(:,t) = cumsum(weight(:,t));
   % estimate the kinematics 
   x_h(:,t) = s(:,:,t)*weight(:,t);   
end
% centralization "recovery"
x_h(1:2,:) = x_h(1:2,:) + mean_pos'*ones(1,M);  
% show the decoding of x and y positions
figure(2);
subplot(2,1,1);
plot(1:M, kin(:,1),'r--','linewidth', 2);
hold on;
plot(1:M, x_h(1,:)','b--','linewidth', 2);
ylabel('x-position','FontSize', 10);
legend('True','Reconstructed','Location','best');
s1=subplot(2,1,1);
title(s1,'Reconstruction of SMC Algorithm, x-position (K=20)', 'Fontsize', 12);
subplot(2,1,2);
plot(1:M, kin(:,2),'r--','linewidth', 2);
hold on;
plot(1:M, x_h(2,:)','b--','linewidth', 2);
ylabel('y-position','FontSize', 10);
legend('True','Reconstructed','Location','best');
s2=subplot(2,1,2);
title(s2,'Reconstruction of SMC Algorithm, y-position (K=20)', 'Fontsize', 12);
xlabel('time (t)');

%R^2 Error 
r2 = 1 - sum((x_h(1:2,:)'-kin(:,1:2)).^2) ./ ...
                 sum((kin(:,1:2)-ones(M,1)*mean_pos).^2);  
disp(['R2 = ' num2str(r2)]);

