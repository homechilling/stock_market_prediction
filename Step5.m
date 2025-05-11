%% Step 5

%% Preparation
close all;
clear all;
clc;

data = rd_feat('step3.dat');
feat = data(:,9:13); % 5 PCA features
ndays = length(data(:,1));

data = rd_feat('step2.dat');
htar = data(:,9);% high targets
ltar = data(:,10); % low targets
hstate = data(:,12);% target states for highs
lstate = data(:,13);% target states for lows


%% Pre-processing
% Note following issues:
% 1. Data for last 10 days can't be used because there's no targets
% 2. Data for first 49 days can't be used because there's no features.
% 3. Data for day 50~69 should be deleted because first indicators for
% first twenty days are not correct
% 4, Sliding block method will be used. More days should be given up before
% NN training starts.

ndays_tr = 261; % Number of trainning days -- one year
ndays_test = 22; % Number of testing days -- two month
target_bad_days = 10;
feature_bad_days = 49;
indicator_bad_days = 20;

good_days = floor((ndays-ndays_tr-target_bad_days-feature_bad_days ...
    -indicator_bad_days)/ndays_test)*ndays_test;
%% Neural Networks Setting
Nnodes = 10;
hnet = feedforwardnet(Nnodes,'traincgf'); % NN for highs
% Conjugate gradient backpropagation with Fletcher-Reeves updates
hnet.trainParam.epochs = 18;

hnet.divideParam.trainRatio = 0.6;
hnet.divideParam.valRatio = 0;
hnet.divideParam.testRatio = 0;
% ratios are set as recommended by internet threads
hnet.inputs{1}.processFcns={};
hnet.outputs{2}.processFcns={};
% input/output processing functions are as default

hnet.performFcn='sse';
% Sum squared error performance function
hnet.layers{1}.transferFCN='tansig';
% Hyperbolic tangent sigmoid transfer function


lnet = feedforwardnet(Nnodes,'traincgf'); % NN for lows
lnet.trainParam.epochs = 18;
lnet.divideParam.trainRatio = 0.6;
lnet.divideParam.valRatio = 0;
lnet.divideParam.testRatio = 0;
lnet.inputs{1}.processFcns={};
lnet.outputs{2}.processFcns={};
lnet.performFcn='sse';
lnet.layers{1}.transferFCN='tansig';
%% Neural Networks Trainning/Testing in Sliding Block Manner 
% The training and testing of two NNs can be done in one loop, but the 
% demo won't be clear. So two loops were created; high NN first, low NN
% second.
output = zeros(ndays,2);
for i = 50:ndays_test:good_days
[hnet tr]=train(hnet,feat(i:i+ndays_tr-1,:)',htar(i:i+ndays_tr-1,:)'); 
htrain = sim(hnet,feat(i:i+ndays_tr-1,:)');% trainning
hpredict = sim(hnet,feat(i+ndays_tr:i+ndays_tr+ndays_test-1,:)'); %testing

% prepare for output
output(i+ndays_tr:i+ndays_tr+ndays_test-1,1) = hpredict;

% plot the trainning/testing data for a quick look
figure(1)
plot(htar(i:i+ndays_tr-1,:),'g')
hold on
plot(htrain,'r')
hold off
legend('Target','Train')
title('Trainning')

figure(2)
plot(htar(i+ndays_tr:i+ndays_tr+ndays_test-1),'g')
hold on
plot(hpredict,'r')
hold off
legend('Target','Train')
title('Testing')
end
disp('NN for highs part is done')
pause
for i = 50:ndays_test:good_days
[lnet tr]=train(lnet,feat(i:i+ndays_tr-1,:)',ltar(i:i+ndays_tr-1,:)'); 
ltrain = sim(lnet,feat(i:i+ndays_tr-1,:)');% trainning
lpredict = sim(lnet,feat(i+ndays_tr:i+ndays_tr+ndays_test-1,:)'); %testing

% prepare for output
output(i+ndays_tr:i+ndays_tr+ndays_test-1,2) = lpredict;

% plot the trainning/testing data for a quick look
figure(1)
plot(ltar(i:i+ndays_tr-1,:),'g')
hold on
plot(ltrain,'r')
hold off
legend('Target','Train')
title('Trainning')

figure(2)
plot(ltar(i+ndays_tr:i+ndays_tr+ndays_test-1),'g')
hold on
plot(lpredict,'r')
hold off
legend('Target','Train')
title('Testing')
end
disp('NN for lows part is done')

%% Output Data
output = [data output data(:,11) ];
wr_feat(output,'step5.dat');