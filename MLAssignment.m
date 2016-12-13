clear all; close all; clc;
%--------------------------------------------------------------------------
% ASSIGNMENT    % Student: Nadezhda Gracheva      email: ng5n15  
%                      ID: 28919718
%                          System, control and signal prcessing
%--------------------------------------------------------------------------
%% Part 1.  Neural Network Approximation

m1=[0 3]'; m2=[2 1]';                            % Distinct means
C1=[2 1; 1 2]; C2=[1 0; 0 1];                    % Distinct covariances
         
numGrid = 50;
xRange = linspace(-4.0, 8.0, numGrid);   
yRange = linspace(-4.0, 8.0, numGrid);
P1 = zeros(numGrid, numGrid);                   % control of the memory used
P2 = P1;
for i=1:numGrid
  for j=1:numGrid
  x = [yRange(j) xRange(i)]';
  P1(i,j) = mvnpdf(x', m1', C1);                 %Create data matrixes
  P2(i,j) = mvnpdf(x', m2', C2);
  end
end
%______Plotting data_________ 
P1max = max(max(P1));
P2max = max(max(P2));
 figure(1);
contour(xRange,yRange, P1,[0.1*P1max 0.5*P1max 0.8*P1max],'LineWidth',2);
hold on
plot(m1(1), m1(2), 'b*','LineWidth',4);
contour(xRange,yRange, P2, [0.1*P2max 0.5*P2max 0.8*P2max],'LineWidth',2);
plot(m2(1), m2(2), 'r*','LineWidth',4);
title('Two Gaussian distributed classes of data','FontSize', 14);
%______ Plotting a particular  number of samples______ 
N = 100;                                       % N - is a number of samples
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);
plot(X1(:,1),X1(:,2),'bx', X2(:,1),X2(:,2),'rs');grid on
%______Plotting a decision boundry for posterior probability = 0.5 ____
labels=[ones(100,1);-1*ones(100,1)];                %Lable our classes
minvalue=min(min([X1 X2]));                         % calculate min and max values to build coordinates
maxvalue=max(max([X1 X2]));
[xx1, xx2]=meshgrid(minvalue:.1:maxvalue,minvalue:.1:maxvalue);   % creat coordinates
pointspace=[xx1(:) xx2(:)];                                       % all the points of our coordinates

naive=NaiveBayes.fit([X1;X2],labels);               % Builds a naive bayes classifier

P = posterior(naive,pointspace);
graphsize=size(xx1);
probpoints=[pointspace P(:,1)];                     % Probability changes while moving along the grid 

 figure(2);
surf(xx1,xx2,reshape(probpoints(:,3),graphsize))    % 3D postetior prob (the other method)
  figure(3);
[rowProb,~]=size(probpoints);
for i =1: rowProb
    if probpoints(i,3) < 0.5
        hold on
        plot(probpoints(i,1),probpoints(i,2),'cx')   %colours he part which probobility is lower than a half
    else 
        hold on
        plot(probpoints(i,1),probpoints(i,2),'yo')
    end
end

hold on;
plot(X1(:,1),X1(:,2),'ko');           % put samples and boundry for the posterior prob together
hold on;
plot(X2(:,1),X2(:,2),'mx');
     title('The decision boundary for P[w1|x]=0.5','FontSize', 18);
%
%_____ Plotting the 3D posterior probability for each class___________
PP1=P1./(P1+P2); 
PP2=P2./(P1+P2);
   figure(4)
mesh(xRange,yRange,PP1);           % 3D posterior probability for the first class
title('3D graph of the posterior porbability of the first class','FontSize', 18);
   figure(5)
mesh(xRange,yRange,PP2);          % 3D posterior probability for second class
hold on;
plot(X1(:,1),X1(:,2),'ko');           % put samples and boundry for the posterior prob together
hold on;
plot(X2(:,1),X2(:,2),'mx');
title('3D graph of the posterior porbability of the second class','FontSize', 18);

% A feedforward neural network
   
   x=[X1;X2]';                             % prepare data for training
   t=[ones(N,1); -1*ones(N,1)]';           % lable data for training
   tic
net = feedforwardnet(10);                                    
net = train(net, x, t);                         
output = net(x);
%view(net)
   toc
perf=perform(net,t',output);               % 
  [~,n2]=size(output);
  
  X11 = mvnrnd(m1, C1, N);
X22 = mvnrnd(m2, C2, N);
 xts=[X11;X22]';
 output = net(xts);
  figure(6)
  for i=1:n2
      if output(:,i)>0
          hold on
          plot(x(1,i),x(2,i),'cx')
      else
          hold on
          plot(x(1,i),x(2,i),'bo')
      end
  end
  
 
% generate a grid
span = -4:.05:8;
[P1,P2] = meshgrid(span,span);                 %%!!!!! Try to play with shifting classes farer from each other to show the clear boundary
pp = [P1(:) P2(:)]';
% simulate neural network on a grid
aa = net(pp);
% translate output into [-1,1]
%aa = -1 + 2*(aa>0);
% plot classification regions
  figure(6)
mesh(P1,P2,reshape(aa,length(span),length(span))-5);
colormap jet

hold on;
 plot(X1(:,1),X1(:,2),'bd');           %Samples of data to compare with boundary
 hold on;
 plot(X2(:,1),X2(:,2),'m*');
 title('Neural network desicion contour ','FontSize', 15);
%% Part 2.     Time Series Prediction

% input arguments
a        = 0.2;     % value for a in eq (1)
b        = 0.1;     % value for b in eq (1)
tau      = 17;		% delay constant in eq (1)
x0       = 1.2;		% initial condition: x(t=0)=x0
deltat   = 1;	    % time step size (which coincides with the integration step)
sample_n = 2000;	% total no. of samples, excluding the given initial condition
interval = 1;	    % output is printed at every 'interval' time steps

% x_t : x at instant t , i.e. x(t) (current value of x)
% x_t_minus_tau : x at instant (t-tau) , i.e. x(t-tau)
% x_t_plus_deltat : x at instant (t+deltat), i.e. x(t+deltat) (next value of x)
% X : the (sample_n+1)-dimensional vector containing x0 plus all other computed values of x
% T : the (sample_n+1)-dimensional vector containing time samples
% x_history : a circular vector storing all computed samples within x(t-tau) and x(t)
% ___________________________________________________________________________
% Main algorithm
time = 0;
index = 1;
history_length = floor(tau/deltat);
x_history = zeros(history_length, 1); % here we assume x(t)=0 for -tau <= t < 0
x_t = x0;

X = zeros(sample_n+1, 1); % vector of all generated x samples
T = zeros(sample_n+1, 1); % vector of time samples

for i = 1:sample_n+1
    X(i) = x_t;
    if (mod(i-1, interval) == 0)
%         disp(sprintf('%4d %f', (i-1)/interval, x_t));
    end
    if tau == 0
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history(index);
    end

    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b);

    if (tau ~= 0)
        x_history(index) = x_t_plus_deltat;
        index = mod(index, history_length)+1;
    end
    time = time + deltat;
    T(i) = time;
    x_t = x_t_plus_deltat;
end


figure(7)
plot(T, X);
set(gca,'xlim',[0, T(end)]);
xlabel('t');
ylabel('x(t)');
title(sprintf('A Mackey-Glass time serie (tau=%d)', tau));
 % Design training set and train weights
Ntr=1500;             % number of training samples
p=20;
A=zeros(Ntr-p+1,p);   % A - design matrix

for i=1:p
    for j = 1: Ntr-p+1
       A(j,i)=X(j+(i-1));          % Every row is a time shifted version of a previous one
    end 
end
f=zeros(Ntr-p+1,1);
for i =1:Ntr-p+1
      f(i)=X(p+i);                 % target
end
w = inv(A'*A)*A'*f;
w=A\f;                             % train weights                        
 pred_tr=A*w;                      % prediction on training data;
% Plotting results__________
 figure(8),
 Ttr=zeros(Ntr-p+1,1);      
 for i=1:Ntr-p+1
      Ttr(i)=T(i+p);       % time for training data
 end
plot( Ttr,f,'c',Ttr,pred_tr, 'r', 'LineWidth', 2),
figure
plot(f,pred_tr, 'r.', 'LineWidth', 2),
grid on
xlabel('True values', 'FontSize', 14)
ylabel('Prediction', 'FontSize', 14)
title('Linear Regression: tarining data ', 'FontSize', 14)
Etr=mean(A*w-f).^2;
%---------------------------------------------------------------------
%%Design a test set                                      
Xts=zeros(500,1);                         %Prepare test data
for i=1:500 
        Xts(i)=X(Ntr+i);
end
[nr,~]=size(Xts);                  % size of the test data matrix
Ats=zeros(nr-p+1,p);                % create a design matrix for the test data
for i=1:p
    for j = 1:500-19
       Ats(j,i)=Xts(j+(i-1));         % Every row is a time shifted version of a previous one
    end 
end
Xts=zeros(501,1);                         %Prepare test data
for i=1:501 
        Xts(i)=X(Ntr+i);
end
fts=zeros(nr-p+1,1);
for i =1:nr-p+1
      fts(i)=Xts(p+i);                         % test target
      fts_v(i)=Xts(p+i-1);
end    
fts_v=fts_v';
pred_ts=Ats*w;                      % prediction on test data;
% Plotting__________
Tts=zeros(nr-p+1,1);      
 for i=1:nr-p+1
      Tts(i)=T(i+p);       % time for training data
 end
 figure(9),
plot( fts,pred_ts, 'r', 'LineWidth', 2),
grid on
xlabel('True test values', 'FontSize', 14)
ylabel('Prediction on test data', 'FontSize', 14)
title('Linear Regression:test data ', 'FontSize', 18)
figure
plot( 1:481,pred_ts,'r',1:481,X(1521:2001),'b'),
Etest=mean(Ats*w-fts).^2;
% A feedforward neural network and its performance on one step ahead prediction.

 tic
net = feedforwardnet(10);        % ��� �� ��������� ���� ������� ������ ����� ���� � ������� ������������ ����� � ��������                           
net = train(net,A',f');             % net = feedforwardnet([40 40 40 40 40]);  �������� ���. ����� ��������� ����� ����� ���� ��� ����������.           
output = net(A');

view(net)
   toc
perf_series=perform(net,f',output);               % 
    figure(10)
 plot(T, X);
 hold on
 Tts=zeros(nr-p+1,1);      
 for i=1:nr-p+1
      Tts(i)=T(i+p);       % time for training data
 end
 
                            
 output2=net(Ats');            %Nn for test data 
 output2=output2';
 Err_nn_ts=mean(output2-fts).^2;
hold on 
 figure(100)
 plot(1:2001, X(1:2001),'r');
 hold on
 plot(21:1501, output,'c',1521:2001,output2,'b');
 xlabel('time (steps)')
ylabel('Function values')
title('Neural Network performance on one step ahead prediction.','FontSize',18)
% Free running mode.

 
 Ats1=Ats(1,:); % first row of matrix to predict the first row
    output_series_test=zeros(481,1);
 for i=1:481                          
      output_series_test(i)=net(Ats1');            % for test data 
      for n=1:19
          Ats1(n)=Ats1(n+1);
          Ats1(20)=output_series_test(i);
      end
 end
 Err_Free_running=mean(output_series_test-fts).^2
 
 figure(12)
 plot( Tts,fts,'*c',Tts,output_series_test, 'r'),
legend('testing samples','oscillations','Location','northwest')
title('Neural network output in a free running mode.','FontSize',18)

% Financial time series  

x=1:1008;                  %time step;
xtr=20:700;
Xts=701:1000;
%--------------------------------Prepare parameters---------------------
fin_data = xlsread('table.csv');    % Reading the data from an excel file.
[fin_data,txt,raw] = xlsread('table.csv');
volume_financial=fin_data(:,5); 
    close_pr=fin_data(:,4);                  % fourth column - 'close', will be used for formulating a neural net work
for n=1:1008
    close_data(n)=close_pr(1008-n+1);
end
vol_data=zeros(1008,1);
for n=1:1008 
    vol_data(n)=volume_financial(1008-n+1);    %invert the order so that it goes according to time
end

close_pr_train=close_data(1:700);  
close_pr_test=close_data(701:1001);

target_tr=close_data(21:701);
target_test=close_data(721:1001);

vol_data_train=vol_data(21:701)';
vol_data_test=vol_data(721:1001)';



figure(13)
plot(close_data,'c');
title('Financial time series.Close','FontSize',14);

% predictor.
Nf=1008;
Ntrf=700;             % number of training samples
p=20;
Af=zeros(Ntrf-p+1,p);   % A - design matrix

for i=1:p
    for j = 1: Ntrf-p+1
       Af(j,i)=close_data(j+(i-1));          % Every row is a time shifted version of a previous one
    end 
end
ff=zeros(Ntrf-p+1,1);
for i =1:Ntrf-p+1
      ff(i)=close_data(p+i);                 % target
end

% Design a test set                                      
Xtsf=zeros(300,1);                         %Prepare test data
for i=1:300 
        Xtsf(i)=close_data(Ntrf+i);
end
[nr,nc]=size(Xtsf);                  % size of the test data matrix
Atsf=zeros(nr-p+1,p);                % create a design matrix for the test data
for i=1:p
    for j = 1:300-19
       Atsf(j,i)=Xtsf(j+(i-1));         % Every row is a time shifted version of a previous one
    end 
end

ftsf=zeros(nr-p+1,1);
for i =1:nr-p+1
      ftsf(i)=Xtsf(p+i-1);                         % test target
end            
 tic 
net = feedforwardnet(15);                              
net = train(net,Af',ff');                       
output = net(Af');
%view(net)
   toc
   perf_series(i)=perform(net,ff(1)',output);               
    figure(14)
 plot(x,close_data);
 hold on
 plot(xtr,output,'c','LineWidth', 2);
   output2=net(Atsf');            % for test data      
 output2=output2';
 Err_close_finan=mean(output2-ftsf).^2
 figure(15);
 plot(ftsf,'g')
 hold on
 plot(output2, 'r.')
title('Financial time series prediction on 300 test data samples','FontSize',14);
%--------------------------------------------------------------------------
% Using volume to check wether additional information helps to make
% prediction better 
volume_finantial=fin_data(:,5); %reading volume from the data matrix
vol_data=zeros(1008,1);
for n=1:1008 
    vol_data(n)=volume_finantial(1008-n+1);    %invert the order so that it goes according to time
end

% predictor.
A_close_volume=zeros(Ntrf-p+1,2*p);   % A - design matrix with close and volume values

for i=1:2:2*p 
    for n=1:p
        for j = 1: Ntrf-p+1
       A_close_volume(j,i)=close_data(j+(n-1));        % design of a matrix which contains A(x(n-1),v(n-1),...)
       A_close_volume(j,i+1)=vol_data(j+(n-1));       % Every row is a time shifted version of a previous one
        end 
    end
end

% Design a test set                                      
                 % size of the test data matrix
vol_data_test=zeros(300,1);                         %Prepare test data
for i=1:300 
        vol_data_test(i)=vol_data(Ntrf+i);
end
vol_data_ts=vol_data_test(1:281);
Ats_close_volume=zeros(nr-p+1,2*p);                 % create a design matrix for the test data
for i=1:2:2*p
     for n=1:p
         for j = 1:281
       Ats_close_volume(j,i)=Xtsf(j+(n-1));         % Every row is a time shifted version of a previous one
       Ats_close_volume(j,i+1)=vol_data_test(j+(n-1));   % design a test matrix which contains A(x(n-1),v(n-1),...)
         end 
     end
end
         

net = feedforwardnet(15);                              
net = train(net,A_close_volume',ff');                       
output_c_v = net(A_close_volume');
%view(net)%
%perf_series(i)=perform(net,ff(1)',output);               
    figure(16)
 plot(x,close_data);
 hold on
 plot(xtr,output_c_v,'c','LineWidth', 2);
   output_c_v_ts=net(Ats_close_volume');            % for test data      
 output_c_v_ts=output_c_v_ts';
 Error_c_v_ts=mean(output_c_v_ts-ftsf).^2
 figure(17);
 plot(ftsf,'g')
 hold on
 plot(output_c_v_ts, 'r.')
title('Financial time series prediction on test data samples(300) considering "volume" input','FontSize',14);






