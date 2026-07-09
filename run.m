clear;clc;close all;
addpath('data')
addpath('funs');

dataset_name='prokaryotic';
load([dataset_name,'.mat'])
num=size(X{1},1);
V=length(X);
c=length(unique(Y));

% Normalization
for i = 1 :length(X)
    X{i} = full((X{i} - mean(X{i}, 2)) ./ repmat(std(X{i}, [], 2), 1, size(X{i}, 2)));
    X{i}=X{i}';
end

%% Parameter Setting of prokaryotic with normalization     ACC = 0.8
k=4;	 mu=1;	 beta=1;	 L=3;	 t=7;

%% Parameter Setting of BBC with normalization       ACC = 0.89
% k=5;	 mu=10;	 beta=10;	 L=2;	 t=4;	

%% Parameter Setting of HW without normalization0        ACC = 0.95
% k=20;	 mu=10;	 beta=0.1;	 L=3;	 t=10;

%% Parameter Setting of mnist4 with normalization     ACC = 0.90
% k=8;	 mu=10;	 beta=0.01;	 L=5;	 t=5;

%% Parameter Setting of UCI with normalization               ACC = 0.93
% k=30;	 mu=100;	 beta=1;	 L=4;	 t=11;

%% Optimization of HPS-MVC
[M] = HPSG(X,c,Y);
[F,obj] = main_HPS_MVC(X,beta,mu,k,t,M(1:L),c);
Y_pre=kmeans(F,c,'Replicates',10,'MaxIter',50);
my_result = ClusteringMeasure_new(Y, Y_pre);

disp(['********************************************']);
disp(['Running HPS-MVC on ',dataset_name,' to obtain ACC: ', num2str(my_result.ACC)]);
disp(['********************************************']);

