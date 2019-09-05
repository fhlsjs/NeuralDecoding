clear; close all;

load midterm_train.mat;
[M, N] = size(rate);
c = size(kin, 2);
kin_mean = mean(kin);
kin = kin - ones(M,1)*kin_mean;

rate_mean = mean(rate);

% Estimate A and W using formula;
tic;
a = kin(2:M,:)'; b = kin(1:M-1,:)';
A = a*b'*inv(b*b');   
W = (a-A*b)*(a-A*b)'/(M-1);

% Estimate theta using MLE;

theta = zeros(c+1, N);
X = [ones(M,1) kin]';

% The first and second derivatives;
df1 = inline('X*Y-X*exp(transpose(X)*t)', 'Y', 'X', 't');
df2 = inline('-ones(5,1)*exp(transpose(t)*X).*X*transpose(X)', 't', 'X');

for i = 1:N
    t = theta(:,i)-pinv(df2(theta(:,i),X))*df1(rate(:, i), X, theta(:,i));
    while(norm(t-theta(:,i))>1e-3)
        theta(:,i) = t;
        t = theta(:,i)-pinv(df2(theta(:,i),X))*df1(rate(:, i), X, theta(:,i));
    end

end


% Decoding using point process filter;
load midterm_test.mat;

[M, N] = size(rate);
c = size(kin, 2);

kin_est3 = zeros(c, M);
kin_m = zeros(c, M);
W_k = zeros(c, c, M);
W_m = zeros(c, c, M);
alpha = theta(2:5,:);

for i = 2:M
    W_m(:,:,i) = A*W_k(:,:,i-1)*A'+W;
    kin_m(:,i) = A*kin_est3(:,i-1);
    
    W_k(:,:,i) = inv(inv(W_m(:,:,i))+ones(c,1)*exp([1 kin_m(:,i)']*theta).*alpha*alpha');
    kin_est3(:,i) = kin_m(:,i)+W_k(:,:,i)*alpha*(rate(i,:)-exp([1 kin_m(:,i)']*theta))';
end


kin_est3 = kin_est3 + kin_mean'*ones(1,M);
kin_m = kin_m + kin_mean'*ones(1,M);
t3 = toc;

figure(1);
title('Inhomogeneous Poisson Model', 'Fontsize', 12);
subplot(2,2,1);
plot(1:M, kin(:,1), 'r',1:M, kin_est3(1,:)','b', 'linewidth', 2);
ylabel('X-position', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,2);
plot(1:M, kin(:,2),'r', 1:M, kin_est3(2,:)','b', 'linewidth', 2);
ylabel('Y-position', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,3);
plot(1:M, kin(:,3), 'r',1:M, kin_est3(3,:)','b', 'linewidth', 2);
ylabel('X-velocity', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,4);
plot(1:M, kin(:,4),'r', 1:M, kin_est3(4,:)','b', 'linewidth', 2);
ylabel('Y-velocity', 'FontSize', 11);
legend('True', 'Estimate');


R2_IPP1 = 1-sum((kin - kin_est3').^2)./...
           sum((kin - ones(M,1)*kin_mean).^2);



       
       
clear; close all;

load midterm_train.mat;
[M, N] = size(rate);
c = size(kin, 2);
kin_mean = mean(kin);
kin = kin - ones(M,1)*kin_mean;

rate_mean = mean(rate);

% Estimate A and W using formula;
tic;
a = kin(2:M,:)'; b = kin(1:M-1,:)';
A = a*b'*inv(b*b');   
W = (a-A*b)*(a-A*b)'/(M-1);

% Estimate theta using MLE;

theta = zeros(c+1, N);
X = [ones(M,1) kin]';

% The first and second derivatives;
df1 = inline('X*Y-X*exp(transpose(X)*t)', 'Y', 'X', 't');
df2 = inline('-ones(5,1)*exp(transpose(t)*X).*X*transpose(X)', 't', 'X');

for i = 1:N
    t = theta(:,i)-pinv(df2(theta(:,i),X))*df1(rate(:, i), X, theta(:,i));
    while(norm(t-theta(:,i))>1e-3)
        theta(:,i) = t;
        t = theta(:,i)-pinv(df2(theta(:,i),X))*df1(rate(:, i), X, theta(:,i));
    end

end
      
       
% Below is the sequential monte carlo method;
load midterm_test.mat;

[M, N] = size(rate);
c = size(kin, 2);
K = 500; % sample size;
s = zeros(c, K, M);
weight = ones(K, M)/K;
Weight = cumsum(weight);
kin_est4 = zeros(c, M);

ll = zeros(K,1);

for t = 2:M
    if mod(t,100) ==0
        disp(sprintf('t = %d', t));
    end
    
    r = rand(1,K);
    j = sum(Weight(:,t-1)*ones(1,K) < ones(K,1)*r)+1;
    s_p = s(:,j,t-1);
    s(:,:,t) = A*s_p + mvnrnd(zeros(1,c),W,K)';
        % calculate log-likelihood;
    for k = 1:K
        ll(k,1) = rate(t,:)*theta'*[1;s(:,k,t)]-sum(exp(theta'*[1;s(:,k,t)]));
        
    end
    
    % weight at step t;
    weight(:,t) = ones(K,1)./sum(exp(ones(K,1)*ll'-ll*ones(1,K)),2);
    Weight(:,t) = cumsum(weight(:,t));
    
    % estimate the kinematics;
    kin_est4(:,t) = s(:,:,t)*weight(:,t);
end

kin_est4 = kin_est4 + kin_mean'*ones(1,M);
t4 = toc;

figure(2);

title('Inhomogeneous Poisson Model', 'Fontsize', 12);
subplot(2,2,1);
plot(1:M, kin(:,1), 'r',1:M, kin_est4(1,:)','b', 'linewidth', 2);
ylabel('X-position', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,2);
plot(1:M, kin(:,2),'r', 1:M, kin_est4(2,:)','b', 'linewidth', 2);
ylabel('Y-position', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,3);
plot(1:M, kin(:,3), 'r',1:M, kin_est4(3,:)','b', 'linewidth', 2);
ylabel('X-velocity', 'FontSize', 11);
legend('True', 'Estimate');

subplot(2,2,4);
plot(1:M, kin(:,4),'r', 1:M, kin_est4(4,:)','b', 'linewidth', 2);
ylabel('Y-velocity', 'FontSize', 11);
legend('True', 'Estimate');


R2_IPP2(4,:) = 1-sum((kin-kin_est4').^2)./...
           sum((kin-ones(M,1)*kin_mean).^2);

       
figure(3);   
subplot(2,2,1);
plot(R2_IPP2(:,1),'d-', 'linewidth', 2);
hold on;
plot(1:4,ones(4)*R2_IPP1(1),'-r','linewidth', 2);
title('R-square of X-pos estimation', 'Fontsize', 12);
ylabel('R-square', 'FontSize', 11);

subplot(2,2,2);
plot(R2_IPP2(:,2),'d-', 'linewidth', 2);
hold on;
plot(1:4,ones(4)*R2_IPP1(2),'-r','linewidth', 2);
title('R-square of Y-pos estimation', 'Fontsize', 12);

subplot(2,2,3);
plot(R2_IPP2(:,3),'d-', 'linewidth', 2);
hold on;
plot(1:4,ones(4)*R2_IPP1(3),'-r','linewidth', 2);
title('R-square of X-v estimation', 'Fontsize', 12);
ylabel('R-square', 'FontSize', 11);

subplot(2,2,4);
plot(R2_IPP2(:,4),'d-', 'linewidth', 2);
hold on;
plot(1:4,ones(4)*R2_IPP1(4),'-r','linewidth', 2);
title('R-square of Y-v estimation', 'Fontsize', 12);