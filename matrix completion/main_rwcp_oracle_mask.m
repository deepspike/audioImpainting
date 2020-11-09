%% Our method (demo)
%% The Reference:
%% Qi Liu and Jibin Wu, 
%% "Parameter tuning-free missing-feature reconstruction for robust sound recognition", 
%% IEEE Journal of Selected Topics in Signal Processing, 2020.
%% If there are any problems, please do not hesitate to contact us.
%% If you feel the code is useful, please cite this paper in your work. Thanks.

load('RWCP_test.mat', 'FBETestList');
clc

per = 0.8; % pecentage entries left.
[L,~]= size(FBETestList);
Y_test= cell(L,1);
Y_Omega_test = cell(L,1);
MSE_test = cell(L,1);
SNR_test = zeros(L,1);
t_ours = zeros(L,1);

parfor i = 1:L
    i
    % Our Algorithm
    z = FBETestList{i,1};
    x_T = z; 
    [m1,n1]=size(x_T);
    maxiter = 50;
    array_Omega = binornd( 1, per, [m1, n1] );
    x_Omega = x_T.* array_Omega;
    Y_Omega_test{i} = x_Omega;
    tic
    [x_reconstruct, MSE] = GP2(x_T, x_Omega, maxiter, array_Omega, 2);
    t_ours(i)=toc;
    MSE_test{i} = MSE;
    Y_test{i} = x_reconstruct;
    SNR_test(i) = 20*log(norm(x_T, 'fro')/norm(x_T-x_reconstruct, 'fro'));
end

fprintf('Our Algorithm SNR: %f CPU Time: %f\n', mean(SNR_test), mean(t_ours));