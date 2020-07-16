clear; close all; clc;
tic
%% Release actions according to the states

%% OFFLINE FIRST RUN << << <<

%Run the Simulink Flowframe Simulation
fprintf('//INITIALIZING// Simulink OnlineTry_R2019a_NO_AEB\n');
model = 'OnlineTry_R2019a_NO_AEB';
load_system(model);
simOut = sim(model);
fprintf('//DONE// Simulink OnlineTry_R2019a_NO_AEB\n');

data_processing_simul_NO_AEB(); %It processes and creates the simul_NO_AEB_treated.mat
fprintf('\n//DONE// Function data_processing_simul_NO_AEB\n\n');

%% To develop boundaries in order to keep the car always in the central lane
TargetVelocity=27.7777; % Desired Velocity (>> m/s <<)

%Hyper--Parameters
stnu = 4;               % number of states 
tau = 1;                % Target update
gamma = 0.9;            % discount factor
minipcent= 1;         % Minibatch size

% Convengence parameter
hmin = 25;              % number of training cycles
acte = 0.0001;           % Max squared error between to consecutives Q evaluations
acterror = acte;
hmax = 50;
hcount=0;

simCount=1;             %In the first Run, simCount=1, to use the simul_NO_AEB_treated.mat

%% Data treatment 
[DATA,leng]=Simulation_treatment(TargetVelocity,stnu,simCount);
%simOut = sim(model);

%% Main Algorithm
fprintf('\n//CREATION OF THE NETWORKS// Actor and Critic\n\n');
crtb=zero_critic(stnu+1);                                   % Plus two desired actions
actb=zero_actor(stnu);                                      % Create the 0 actor and critic 

% Minibatch = Select_data(DATA,actb,crtb,gamma,minipcent);  % Select the data and calculates the Q value 

Minibatch = Select_data(DATA,stnu,actb,crtb,gamma,minipcent);%Select the data and calculates the Q value 
fprintf('->->->Minibatch done!\n');
crta = create_critic(Minibatch,stnu+1);                     % Create and train the critic network
fprintf('->->->Critic Network Creation done!\n');
acta = create_actor(Minibatch,crta);                        % Choose the best action, create and train the actor
fprintf('->->->Actor Network Creation done!\n');

save('act.mat','acta');

actb = target_update(actb,acta,tau); % Update the weights of the actor network
crtb = target_update(crtb,crta,tau); % Update the weights of the critic network

if stnu == 4
    actbb = actb([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
    Qb = crtb([actbb;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
    actaa = acta([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
    Qa = crta([actaa;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);    
else
    fprintf('some value has been inputed wrongly');
end

error = gsubtract(actaa,actbb); % Calculate difference between Q
[s,n] = sumsqr(error);
acterror(hcount+1)=s/n;
hv(hcount+1)=hcount;            % Create a vector of the h times that the cicle are tried
hcount=hcount+1;                % Increase the counter
plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);

fprintf('\n//FIRST TRAINING NO AEB// Actor, Critic and Best actions\n\n');

while and(or(acterror(hcount)>=acte, hcount<=hmin),hcount<=hmax)
    
    Minibatch = Select_data(DATA,stnu,actb,crtb,gamma,minipcent); %% Select the data and calculates the Q value 
    crta = train_critic(Minibatch,crta); % Train the critic network
    acta = train_actor(Minibatch,crta,acta); % Choose the best action and train the actor
    actb = target_update(actb,acta,tau); % Update the weights of the actor network
    crtb = target_update(crtb,crta,tau); % Update the weights of the critic network
    
    if stnu == 4
        actbb = actb([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
        Qb = crtb([actbb;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
        actaa = acta([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
        Qa = crta([actaa;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);    
    else
        fprintf('some value has been inputed wrongly');
    end

    error = gsubtract(actaa,actbb); % Calculate difference between Q
    [s,n] = sumsqr(error);
    acterror(hcount+1)=s/n;
    hv(hcount+1)=hcount;  % Create a vector of the h times that the cicle are tried
    hcount=hcount+1;  %Increase the counter
        
    plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);

end

fprintf(['\nAction converged after %d epochs\n'],hmin);

save ('act.mat','actb');
count_reward = sum((DATA(leng:end,6))<=0.1);
save('act_count.mat','actb');
mean_reward=mean(DATA(leng:end,6));
save ('act_mean.mat','actb');
reward_error = 0;


%% ONLINE RUN << << << <<
%data_processing_v4_4inputs();

%Run the Simulink Flowframe Simulation
fprintf('\n//INITIALIZING// Simulink OnlineTry_R2019a_WITH_AEB');
model = 'OnlineTry_R2019a_WITH_AEB';
load_system(model);
simCount=1;
simmin=5;      %Number of loops until reach the less than the reward_error

while and(reward_error<=0.1,simCount<simmin) 
    %% Run simulink simulation to update the data
    simCount=simCount+1; % update counter of the update
    
    %Run simulation
    simOut = sim(model);
    fprintf('\n//DONE// Simulink OnlineTry_R2019a_WITH_AEB\n');
    
    TargetVelocity=27.7777; % Desired Velocity (>> m/s <<)
    data_processing_simul_WITH_AEB();
    fprintf('\n//DONE// Function data_processing_simul_WITH_AEB\n\n');
    
    [DATA,leng]=Simulation_treatment(TargetVelocity,stnu,simCount); %Update the Data
    
    %% Main Algorithm
    
    hcount = 1;
    acterror=0;
    acterror(hcount)=1;
    hv=0;
    hcount2=1;
    
    fprintf('\n//RE-TRAINING WITH AEB// Actor, Critic and Best actions\n\n');

    while and(or(acterror(hcount)>=acte, hcount<=hmin),hcount<=hmax)
        
        Minibatch = Select_data(DATA,stnu,actb,crtb,gamma,minipcent);%Select the data and calculates the Q value 
        crta = train_critic(Minibatch,crta); % Train the critic network
        acta = train_actor(Minibatch,crta,acta); % Choose the best action and train the actor
        actb = target_update(actb,acta,tau); % Update the weights of the actor network
        crtb = target_update(crtb,crta,tau); % Update the weights of the critic network
        
        if stnu == 4
            actbb = actb([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
            Qb = crtb([actbb;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
            actaa = acta([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);
            Qa = crta([actaa;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)']);    
        else
            fprintf('some value has been inputed wrongly');
        end
        
        if hcount2==1
            hcount= 1;
        elseif hcount2==2
            hcount=hcount+1;  %Increase the counter
        end
        
        error = gsubtract(actaa,actbb); % Calculate difference between Q
        [s,n] = sumsqr(error);
        acterror(hcount)=s/n;
        hv(hcount)=hcount;  % Create a vector of the h times that the cicle are tried
        
        %plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);
        hcount2=2;
        
    end

    save('act.mat','actb');
    fprintf(['\nAction converged after %d epochs\n'],hmin);
    
    %% Run the simulation to evaluate the behavior of the AEB DEEP RL Network
    
    fprintf('\n//Evaluating AEB DEEP RL Network//\n\n');
    
    %Run simulation
    simOut = sim(model);
    
    TargetVelocity=27.7777; % Desired Velocity (>> m/s <<)
    DATA2 = Simulation_treatment2(TargetVelocity,stnu,simCount); % Update the data 
            
    reward_error_vect(simCount-1) = min(DATA2(:,6));
    reward_error = min(DATA2(:,6));
    if reward_error>0.1
        save('act_error.mat','actb');
        fprintf('\nActor update by number of bad rewards = 0\n');
    end
    % Update the saved actor if the mean of rewards is bigger then the old one    
    mean_reward_new = mean(DATA2(:,6));
    mean_rew_vect(simCount-1)=mean_reward_new;
    if mean_reward_new>mean_reward
        mean_reward=mean_reward_new;
        save ('act_mean.mat','actb');
        fprintf('\nActor update by the mean\n');
    end
    
    % Update the saved actor if the number of bad rewards is small then the old one
    count_rew_vect(simCount-1)=sum((DATA2(:,6))<=0.1);
    if count_reward > sum((DATA2(:,6))<=0.1)
        count_reward = sum((DATA2(:,6))<=0.1)
        save('act_count.mat','actb');
        fprintf('\nActor update by number of bad rewards\n');
    end
    
    reward_plot=figure(3);
    subplot(2,1,1)
    plot(reward_error_vect,'-o')
    hold on
    plot(mean_rew_vect,'-o')
    legend('Minimum reward','Medium reward')
    hold off
    
    subplot(2,1,2)
    plot(count_rew_vect,'-o')
    
%%                     DATA2 matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut |Pedal |REWARD| TTC  | dr_ol| vr_ol| v_vut| Pedal|
%|Pedal |      |      |      |      |      |      | next | next | next | next |
end

reward=[reward_error_vect;mean_rew_vect;count_rew_vect];
save ('rewards.mat','reward');
final_time=toc;
saveas(reward_plot,'Reward Plot.png');

fprintf(['\n\nSUMMARY\nNumber of interactions = %d\n',...
         'Time to simulation = %d minutes\n',...
         'Total used data = %d\n',...
         'Number of states = %d\n',...
         'Target Update in the Network update = %f\n',...
         'Discount factor of Q-value = %f\n',...
         'Minibatch size = %d per cent \n',...
         'Maximum mean reward = %f \n' ],simCount,round(final_time/60),length(DATA(:,1)),stnu,round(tau,2),round(gamma,2),minipcent*100,mean_reward);
     
%%
















%% Actor Functions

function net = zero_actor(n)

%Create a actor network with 2 hidden layers with 12 nodes each, and the
% desired number of inputs. Hidden layers with ReLu and output layer
% sigmoid.
%Reset all weights and bias of the actor network to zero.


net=feedforwardnet([38 38]);                % Create the network with 2 hidden layers with 38 nodes each
net.numInputs = n;                          % Define the number of the inputs {dr_ol...vr_ol,GP,BP} 8
%net.numInputs
for i=2:n
    net.inputConnect(1,i) = true;           % Connect all inputs with the first layer
end
net.trainParam.showWindow = 0;              % Turn off the learning pop up as long this learnig is not important
%net.trainParam.showCommandLine=true;
net.trainParam.epochs = 1;                  % Use just one pass through network as long this learnig is not important
for i=1:n
    in{i,1}=randn(1,3);                     % Create a random input cell array {number of input x 1}
end
out = randn(1,3);                           % Create a random output ############

net.inputs{1}.processFcns = {}; 
net.outputs{2}.processFcns = {};
net.outputs{3}.processFcns = {};

% net.layers{1}.transferFcn = 'poslin';     % Change the activation function to ReLu in the hidden layers
% net.layers{2}.transferFcn = 'poslin'; 
% net.layers{3}.transferFcn = 'purelin';    % Change the activation function of the output layer to sigmoid 

fprintf('...Actor weights are reseted\n');
net = train(net,in,out);                    % Train to create the array with correct positions of the weights
m = length(getwb(net));                     % Find the number of weights in the 
net = setwb(net,zeros(m,1));                % Insert zeros in all weights and bias

end
function net = create_actor(Minibatch,crt)

actst = cell2mat(Minibatch{1,1});               % Take action and states vector 
[~,n] = size(actst);                            % Find the number of samples
in = (crt.numInputs - 1);                       % Find the number of inputs 
%fprintf('\nNumber of inputs Actor')
%in
%fprintf(1,'Calculating the best action');

% actst = cell2mat(Minibatch{1,1});             % Take action and states vector 
% n = length(actst(1,1));                       % Find the number of samples
% in = (crt.numInputs - 1);                     % Find the number of inputs 

Q=zeros(21,n);
action=zeros(1,n);
    
max1=-1;   %Value for minimum action
max2=1;    %Value for maximum action
zeroone = linspace(max1,max2,21);
acttest = ones(21,n);
for k = 1:21
    acttest(k,:) = zeroone(k).*ones(1,n);
end

for j=1:2
    for k = 1:21 % find the Q value for all of the input actions
        if in == 4
            Q(k,:)=crt([acttest(k,:);actst(2,:);actst(3,:);actst(4,:);actst(5,:)]);
        end
    end
    for k = 1:n
        [~, ind1(k)] = max(Q(:,k));
        Q(ind1,k) = -Inf;
        [~, ind2(k)] = max(Q(:,k));
        max1(k) = acttest(ind1(k),k);
        max2(k) = acttest(ind2(k),k);
        acttest(:,k)= linspace(max1(k),max2(k),21)'; 
    end
    
end

net=feedforwardnet([38 38]);                % Create the network with 2 hidden layers with 30 nodes each
net.numInputs = in;                         % Define the number of the inputs {ax;i;GP;vx;vw}
%fprintf('\nNumber of inputs Actor')
%in
for i=2:in
    net.inputConnect(1,i) = true;           % Connect all inputs with the first layer
end

% Need to remove the process Functions as long as the inputs and outputs
% are by standart treated by the function mapinmax and the used data was
% already treated an normalized 

%      Function      |                    Algorithm
%--------------------------------------------------------------------------
%      mapminmax     | Normalize inputs/targets to fall in the range [?1, 1]
%       mapstd       | Normalize inputs/targets to have zero mean and unity variance
%     processpca     | Extract principal components from the input vector
%     fixunknowns    | Process unknown inputs
% removeconstantrows | Remove inputs/targets that are constant

net.inputs{1}.processFcns = {}; 
net.outputs{2}.processFcns = {};
net.outputs{3}.processFcns = {};

% net.layers{1}.transferFcn = 'poslin';     % Change the activation function to ReLu in the hidden layers
% net.layers{2}.transferFcn = 'poslin'; 
% net.layers{3}.transferFcn = 'purelin';    % Change the activation function of the output layer to sigmoid 
                                            % as long as just values between 0 and 1 are waited to the output
%net.trainFcn='trainlm';     
net.trainParam.max_fail = 20;               %Maximum validation failures
net.trainParam.min_grad = 1e-7;             %Minimum performance gradient
% net.trainParam.mu	0.001                   %Initial mu
% net.trainParam.mu_dec	0.1                 %mu decrease factor
% net.trainParam.mu_inc	10                  %mu increase factor
net.trainParam.mu_max = 1e10;               %Maximum mu
% net.trainParam.show	25                  %Epochs between displays (NaN for no displays)
net.trainParam.showWindow = 0;              %Turn off the learning pop up as long this learnig is not important
% net.trainParam.showCommandLine = true;	%Generate command-line output
fprintf('...(Create_Actor) Actor training\n');
net = train(net,Minibatch{3,1},max1);       % Train to create the array with correct positions of the weights

% Evaluation plot

x=linspace(0,length(action),length(action));
actionnet=net(Minibatch{3,1});
subplot(3,2,[1,2]);
plot(x,action,x,cell2mat(actionnet));
ylabel('Best Action Taken');
xlabel('Amount of data');
end
function act = train_actor(Minibatch,crt,act)

actst = cell2mat(Minibatch{1,1});  % Take action and states vector 
n = length(actst);                 % Find the number of samples
in = (crt.numInputs - 1);          % Find the number of inputs 

%fprintf('...Calculating the best action(train actor)\n');

% actst = cell2mat(Minibatch{1,1});  % Take action and states vector 
% n = length(actst);                 % Find the number of samples
% in = (crt.numInputs - 1);          % Find the number of inputs 

Q=zeros(21,n);
action=zeros(1,n);
    
max1=-1;   %Value for minimum action
max2=1;    %Value for maximum action       
zeroone = linspace(max1,max2,21);
acttest = ones(21,n);
for k = 1:21 %Total of desired actions, states and next states (without function and rewards)
    acttest(k,:) = zeroone(k).*ones(1,n);
end

for j=1:2
    for k = 1:21 % find the Q value for all of the input actions
        if in == 4 %(6 distaces, 6 velocities, vv, pedals)
            Q(k,:)=crt([acttest(k,:);actst(2,:);actst(3,:);actst(4,:);actst(5,:)]);            
        end
    end
    for k = 1:n
        [~, ind1(k)] = max(Q(:,k));
        Q(ind1,k) = -Inf;
        [~, ind2(k)] = max(Q(:,k));
        max1(k) = acttest(ind1(k),k);
        max2(k) = acttest(ind2(k),k);
        acttest(:,k)= linspace(max1(k),max2(k),21)'; 
    end
end

fprintf('...(Train_Actor) Actor training\n');
act= train(act,Minibatch{3,1},max1); % Train to create the array with correct positions of the weights

% Evaluation plot

x=linspace(0,length(action),length(action));
actionnet=act(Minibatch{3,1});
subplot(3,2,[1,2]);
plot(x,action,x,cell2mat(actionnet));
ylabel('Best Action Taken');
xlabel('Amount of data');
end

%% Critic Functions

function net = zero_critic(n)

%Create a critic network with 2 hidden layers with 38 nodes each, and the
% desired number of inputs. Hidden layers with ReLu and output layer
% sigmoid.
%Reset all weights and bias of the actor network to zero.


net=feedforwardnet([38 38]);                % Create the network with 2 hidden layers with 38 nodes each
net.numInputs = n;                          % Define the number of the inputs {ax;i;GP;vx;vw}
for i=2:n
    net.inputConnect(1,i) = true;           % Connect all inputs with the first layer
end
net.trainParam.showWindow = 0;              % Turn off the learning pop up as long this learnig is not important
%net.trainParam.showCommandLine=true;
net.trainParam.epochs = 1;                  % Use just one pass through network as long this learnig is not important
for i=1:n
    in{i,1}=randn(1,3);                     % Create a random input cell array {number of input x 1}
end
out = randn(1,3);                           % Create a random output #

net.inputs{1}.processFcns = {}; 
net.outputs{2}.processFcns = {};
net.outputs{3}.processFcns = {};

% net.layers{1}.transferFcn = 'poslin';     % Change the activation function to ReLu in the hidden layers
% net.layers{2}.transferFcn = 'poslin'; 
% net.layers{3}.transferFcn = 'purelin';    % Change the activation function of the output layer to sigmoid 

fprintf('...Critic weights are reseted\n');
net = train(net,in,out);                    % Train to create the array with correct positions of the weights
m = length(getwb(net));                     % Find the number of weights in the 
net = setwb(net,zeros(m,1));                % Insert zeros in all weights and bias

end
function net = create_critic(Minibatch,in)

net=feedforwardnet([38 38]);            % Create the network with 2 hidden layers with 38 nodes each
net.numInputs = in;                     % Define the number of the inputs {ax;i;GP;vx;vw}
%fprintf('\nNumber of inputs Critic')
%in
for i=2:in
    net.inputConnect(1,i) = true;       % Connect all inputs with the first layer
end

% Need to remove the process Functions as long as the inputs and outputs
% are by standart treated by the function mapinmax and the used data was
% already treated an normalized 

%      Function      |                    Algorithm
%--------------------------------------------------------------------------
%      mapminmax     | Normalize inputs/targets to fall in the range [?1, 1]
%       mapstd       | Normalize inputs/targets to have zero mean and unity variance
%     processpca     | Extract principal components from the input vector
%     fixunknowns    | Process unknown inputs
% removeconstantrows | Remove inputs/targets that are constant

net.inputs{1}.processFcns = {}; 
net.outputs{2}.processFcns = {};
net.outputs{3}.processFcns = {};

% net.layers{1}.transferFcn = 'poslin';  	% Change the activation function to ReLu in the hidden layers
% net.layers{2}.transferFcn = 'poslin'; 
% net.layers{3}.transferFcn = 'purelin';    % Change the activation function of the output layer to sigmoid 
                                            % as long as just values between 0 and 1 are waited to the output
net.trainFcn='trainlm';   

net.trainParam.max_fail = 20;               %Maximum validation failures
net.trainParam.min_grad = 1e-7;             %Minimum performance gradient
% net.trainParam.mu = 0.001;                %Initial mu
% net.trainParam.mu_dec = 0.1;              %mu decrease factor
% net.trainParam.mu_inc = 10;               %mu increase factor
net.trainParam.mu_max = 1e10;               %Maximum mu
% net.trainParam.show	25                  %Epochs between displays (NaN for no displays)
net.trainParam.showWindow = 0;              %Turn off the learning pop up as long this learnig is not important
% net.trainParam.showCommandLine = true;	%Generate command-line output

%        Minibatch Array
%|   {1,1}   | {2,1} | {3,1} |
%|actionstate| Qvalue| state |

fprintf('...(Create_Critic) Critic training\n');
net = train(net,Minibatch{1,1},Minibatch{2,1}); % Train to create the array with correct positions of the weights
end
function crt = train_critic(Minibatch,crt)

fprintf('...(Train_Critic) Critic training\n');
crt = train(crt,Minibatch{1,1},Minibatch{2,1}); % Train to create the array with correct positions of the weights

end

%% Root Functions

function Minibatch = Select_data(DATA,netin,act,crt,gamma,minicent)
% function Minibatch = Select_data(DATA,act,crt,gamma,minicent)
% netin=act.numInputs;

% Select a minibatch from the total data and calculates the Q value

[m,~]=size(DATA);           % take the DATA size
minisize=round(m*minicent); % Minibatch size = minipcent % of the total DATA size
idx = randperm(m,minisize); % create a random position vector
rawbatch = DATA(idx,:);     % Take the desired data from DATA to Minibatch 

rawbatch = rawbatch';

% Create training data structure with vehicle speed
if netin==4 %States cosidered (6 distaces, 6 velocities, vv, pedals)
    state={rawbatch(2,:);rawbatch(3,:);rawbatch(4,:);rawbatch(5,:)};
    nextstate={rawbatch(8,:);rawbatch(9,:);rawbatch(10,:);rawbatch(11,:)};
    actionstate={rawbatch(1,:);rawbatch(2,:);rawbatch(3,:);rawbatch(4,:);rawbatch(5,:)};
    Qvalue= rawbatch(06,:);
%     action=act(nextstate);
%     critic = crt({action{1,1};rawbatch(19,:);rawbatch(20,:);rawbatch(21,:);rawbatch(22,:);rawbatch(23,:);rawbatch(24,:);rawbatch(25,:);rawbatch(26,:);rawbatch(27,:);rawbatch(28,:);rawbatch(29,:);rawbatch(30,:);rawbatch(31,:);rawbatch(32,:)});
%     Qvalue= rawbatch(18,:);%+ gamma.*critic{1,1};
else
    fprintf('\n### Wrong input value ###');
    brake
end

Minibatch={actionstate;Qvalue;state};

%        Minibatch Array
%|   {1,1}   | {2,1} | {3,1} |
%|actionstate| Qvalue| state |

%%                     rawdata matrix (loaded DATA.mat)

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut |Pedal |REWARD| TTC  | dr_ol| vr_ol| v_vut| Pedal|
%|Pedal |      |      |      |      |      |      | next | next | next | next |

end
function [DATA,leng]=Simulation_treatment(TargetVelocity,stnu,simCount)

if simCount == 1
    load('simul_NO_AEB_treated.mat');
else
    load('simul_WITH_AEB_treated.mat');
end

rawdata=stateaction.Data; 
%|(:,1)Des.pedal(:,2)dr_ol(:,3)vr_ol(:,4)v_vut(:,5)Pedal

%% Maximum values 
%It finds the maximun values in the rawdata matrix to normalize it after

MaxDrx   = 0; %Radar range   (m)   #150
MaxVrx   = 0; %Max. velocity (mph) #50

[sizei,sizej]=size(rawdata);

%Normalizes the values between 0 and 1 to improve the learning performance
for j=1:sizej
    if j==2 %Columns of distance values
        MaxDrx=max(abs(rawdata(:,j))); %Save maximum range(m)
        if (MaxDrx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxDrx;
        end
    elseif and(j>=3,j<=4)%Columns of velocity values
        MaxVrx=max(abs(rawdata(:,j))); %Save maximum velocity(mph)
        if (MaxVrx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxVrx;
        end
    end
end

%% Data treatment
%Normalizes the values between 0 and 1 to improve the learning performance

% for i=1:sizej
%     if and(i>=2,i<=4)%Columns of distance values
%         rawdata(:,i) = rawdata(:,i)./MaxDrx;
%     elseif and(i>=5,i<=8)%Columns of velocity values
%         rawdata(:,i) = rawdata(:,i)./MaxVrx;
%     end
% end

leng = length(rawdata(:,1))-1;

rawdata=calculate_reward(rawdata,TargetVelocity,stnu);

%%                     rawdata matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut | TTC  | dr_ol| vr_ol| v_vut| Pedal|REWARD| TTC  |
%|Pedal |      |      |      |      | next | next | next | next |REWARD| TTC  |

% To Export Data Structure
DATA=rawdata(1:leng,[1:5 10 11 6:9]);
save('DATA.mat','DATA');
fprintf('DATA Updated. First Rewards Assigned!\n');

%%                     DATA matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut |Pedal |REWARD| TTC  | dr_ol| vr_ol| v_vut| Pedal|
%|Pedal |      |      |      |      |      |      | next | next | next | next |
end
function rawdata   = calculate_reward(rawdata,TargetVelocity,stnu)

leng = length(rawdata(:,1))-1;

%Copy the next states to the same line that the last state 
for i=1:leng
    for j=1:stnu
        if (i==leng)
            rawdata(i,(j+(stnu+1)))=rawdata(i,(j+1));
        else
            rawdata(i,(j+(stnu+1)))=rawdata(i+1,(j+1));
        end
    end
end
rawdata(:,10)=ones(); %future rewards(10) -> (06)
rawdata(:,11)=ones(); %future TTC    (11) -> (07)
[sizei,sizej]=size(rawdata);

%% #### Reward THIS IS THE BIGGEST CHALLENGE HERE!!! #### %%

%% TTC-based Reward ('Time To Collision' against leading car) and TTS-Based Reward ('Time To Stop' VUT)
RewardDelT=0;
for i=1:leng % Calculate TTC at the next step and save in rawdata(:,33)
    dsafezone=rawdata(i,10)./4;
    TTC=time_to_collision(rawdata(i,8),rawdata(i,9),dsafezone); %Generate TTC for lane O
    TTS=(0.0001.* mpower(rawdata(i,10),2)+0.0733.*rawdata(i,10)+0.6324); %Time for total stop %Fitted for currently car
    DelT=TTC./TTS; %Time to collision compared to time for total stop
%     if (TTC>0) %Positive Time to Collision
%         if and(rawdata(i,16)>dsafezone,rawdata(i,22)>0)      %No collision/Leading vehicle moving away//Best situation
%             RewardTTC=0.5;
%         elseif and(rawdata(i,16)<dsafezone,rawdata(i,22)<0)  %Collision/Leading vehicle approaching//Worst situation
%             RewardTTC=0;
%         end       
%     elseif (TTC<=0) %Negative Time to Collision
%         if and(rawdata(i,16)>dsafezone,rawdata(i,22)<0)      %No collision/Leading vehicle approaching
%             RewardTTC=0.4;
%         elseif and(rawdata(i,16)<dsafezone,rawdata(i,22)>0)  %Collision/Leading vehicle moving away
%             RewardTTC=0;
%         end
%    rawdata(i,31)=TTC;
    if or((DelT>1),(DelT<0)) %Best situation, when TTC is bigger than TTS or when TTC is negative; DelT greater than 1 and less than 0
        RewardDelT=0.5; 
    else
        %For situations where TTC is shorter and there will be collision; DelT between 0 and 1
        RewardDelT=0; 
    rawdata(i,10)=RewardDelT;
    rawdata(i,11)=TTC;
    end
end
%% v_vut-Based Reward (Real and Desired Velocities) 
RewardDelV=0;
for i= 1:leng
    DelV=abs(((minus(rawdata(i,10),TargetVelocity))./TargetVelocity));
    if DelV==0                              %Real velocity v_vut is equal to Desired velocity TargetVelocity
        RewardDelV=0.5;
    elseif and(DelV>0,DelV<=0.1)            %Error between real and desired velocities at 0-10%
        RewardDelV=0.4;
    elseif and(DelV>0.1,DelV<=0.2)          %Error between real and desired velocities at 10-20%
        RewardDelV=0.25;
    elseif DelV>0.2                         %Error between real and desired velocities greater than 20%
        RewardDelV=0;
    rawdata(i,10)=rawdata(i,10)+RewardDelV; %SUM with last reward
    end
end
end
function netNew    = target_update(netNew,netOld,tau) %b  a 
%Calculate the target upgade in the networks
%Inputs are the copy network, the trained network and the update target
netNew = setwb(netNew,(tau.*getwb(netOld)+(1-tau).*getwb(netNew)));
end
function TTC       = time_to_collision(deltaX,deltaV,safe)  %It's considering only the same lane (scalar finction)
X=minus(deltaX,safe); %deltaX is the distance between VUT and leading vehicle.
if deltaV==0
    TTC=0;              %IMPROVE HERE ##### When deltaV==0 ... TTC->Inf
else
    TTC=X./deltaV; %deltaV is the relative velocity between VUT and leading vehicle.
end
end

function DATA2 = Simulation_treatment2(TargetVelocity,stnu,simCount)

load('simul_WITH_AEB_treated.mat');

rawdata=stateaction.Data; 
%|(:,1)Des.pedal(:,2)dr_ol(:,3)vr_ol(:,4)v_vut(:,5)Pedal

%% Maximum values 
%It finds the maximun values in the rawdata matrix to normalize it after

MaxDrx   = 0; %Radar range   (m)   #150
MaxVrx   = 0; %Max. velocity (mph) #50

[sizei,sizej]=size(rawdata);

%Normalizes the values between 0 and 1 to improve the learning performance
for j=1:sizej
    if j==2 %Columns of distance values
        MaxDrx=max(abs(rawdata(:,j))); %Save maximum range(m)
        if (MaxDrx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxDrx;
        end
    elseif and(j>=3,j<=4)%Columns of velocity values
        MaxVrx=max(abs(rawdata(:,j))); %Save maximum velocity(mph)
        if (MaxVrx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxVrx;
        end
    end
end

%% Data treatment
%Normalizes the values between 0 and 1 to improve the learning performance

% for i=1:sizej
%     if and(i>=2,i<=4)%Columns of distance values
%         rawdata(:,i) = rawdata(:,i)./MaxDrx;
%     elseif and(i>=5,i<=8)%Columns of velocity values
%         rawdata(:,i) = rawdata(:,i)./MaxVrx;
%     end
% end

leng = length(rawdata(:,1))-1;

rawdata=calculate_reward(rawdata,TargetVelocity,stnu);

%%                     rawdata matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut | TTC  | dr_ol| vr_ol| v_vut| Pedal|REWARD| TTC  |
%|Pedal |      |      |      |      | next | next | next | next |REWARD| TTC  |

% To Export Data Structure
DATA2=rawdata(1:leng,[1:5 10 11 6:9]);
save('DATA2.mat','DATA2');
fprintf('DATA2 Updated. First Rewards Assigned!\n');

%%                     DATA2 matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |(:,8) ||(:,9)|(:,10)|(:,11)|
%|Des.  | dr_ol| vr_ol|v_vut |Pedal |REWARD| TTC  | dr_ol| vr_ol| v_vut| Pedal|
%|Pedal |      |      |      |      |      |      | next | next | next | next |
end

%% Plot Functions

function plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);

figure (1);
subplot(3,2,3)
plot(Qb,'b');
hold on
subplot(3,2,3)
plot(Qa,'r');
legend('Updated','Calculated');
ylabel('Q-value');
xlabel('Amount of data');
hold off

subplot(3,2,4)
plot(actbb,'b');
hold on
subplot(3,2,4)
plot(actaa ,'r');
legend('Updated','Calculated');
ylabel('Action');
xlabel('Amount of data');
hold off

subplot(3,2,[5,6]);
plot(hv,acterror,'-o');  %Plot Q error convergence
xlabel('Training cycle');
ylabel('Action error');

end
