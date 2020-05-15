clear; close all; clc;
%% Release actions according to the states
%% To develop boundaries in order to keep the car always in the central lane

TVL=22;                 % Desired Velocity (m/s)
safezone=2;             % Safety zone ahead of the VUT with the witdh of the lane
%IMPROVE HERE #^^#     %In the future, implement some technique like circles

%Hyper--Parameters
stnu = 13;              % number of states 
tau = 1;                % Target update
gamma = 0.9;            % discount factor
minipcent= 0.5;         % Minibatch size

% Convengence parameter
hmin = 20; %number of training cycles
acte = 0.001; % Max squared error between to consecutives Q evaluations
acterror = acte;
hmax = 50;
hcount=0;

%% Data treatment 
[DATA,leng]=Simulation_treatment(safezone,TVL);

%% Main Algorithm
crtb=zero_critic(stnu+1);                                   % Plus two desired actions
actb=zero_actor(stnu);                                      % Create the 0 actor and critic 

% Minibatch = Select_data(DATA,actb,crtb,gamma,minipcent);  % Select the data and calculates the Q value 

fprintf('--->->Minibatch done!\n')
fprintf('...Calculating Actor\n...Calculating Critic\n...Finding Best actions\n')

Minibatch = Select_data(DATA,stnu,actb,crtb,gamma,minipcent);               % Select the data and calculates the Q value 
crta = create_critic(Minibatch,stnu+1);                     % Create and train the critic network
fprintf('--->->Critic Network Creation done!\n')
acta = create_actor(Minibatch,crta);                        % Choose the best action, create and train the actor
fprintf('--->->Actor Network Creation done!\n')

save('act.mat','acta');

actb = target_update(actb,acta,tau); % Update the weights of the actor network
crtb = target_update(crtb,crta,tau); % Update the weights of the critic network

if stnu == 13
    actbb = actb([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
    Qb = crtb([actbb;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
    actaa = acta([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
    Qa = crta([actaa;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);    
else
    fprintf('some value has been inputed wrongly');
end

error = gsubtract(actaa,actbb); % Calculate difference between Q
[s,n] = sumsqr(error);
acterror(hcount+1)=s/n;
hv(hcount+1)=hcount;  % Create a vector of the h times that the cicle are tried
hcount=hcount+1;  %Increase the counter
plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);

while and(or(acterror(hcount)>=acte, hcount<=hmin),hcount<=hmax)
    
    Minibatch = Select_data(DATA,stnu,actb,crtb,gamma,minipcent); %% Select the data and calculates the Q value 
    crta = train_critic(Minibatch,crta); % Train the critic network
    acta = train_actor(Minibatch,crta,acta); % Choose the best action and train the actor
    actb = target_update(actb,acta,tau); % Update the weights of the actor network
    crtb = target_update(crtb,crta,tau); % Update the weights of the critic network
    
    if stnu == 13
        actbb = actb([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
        Qb = crtb([actbb;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
        actaa = acta([DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);
        Qa = crta([actaa;DATA(:,2)';DATA(:,3)';DATA(:,4)';DATA(:,5)';DATA(:,6)';DATA(:,7)';DATA(:,8)';DATA(:,9)';DATA(:,10)';DATA(:,11)';DATA(:,12)';DATA(:,13)';DATA(:,14)']);    
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

fprintf('\nAction converged \n')

save ('act.mat','actb');
%count_reward = sum((DATA(end-(2*leng-1):end,7))<=0.1);
save('act_count.mat','actb');
%mean_reward=mean(DATA(end-(2*leng-1):end,7));
save ('act_mean.mat','actb');
%reward_error = 0;

%%




%% Actor Functions
                                            
function net = zero_actor(n)

%Create a actor network with 2 hidden layers with 12 nodes each, and the
% desired number of inputs. Hidden layers with ReLu and output layer
% sigmoid.
%Reset all weights and bias of the actor network to zero.


net=feedforwardnet([30 30]);                % Create the network with 2 hidden layers with 12 nodes each
net.numInputs = n;                          % Define the number of the inputs {dsx_ol...dvx_ol,GP,brake} 14
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

fprintf('...Actor weights are reseted\n')
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

Q=zeros(27,n);
action=zeros(1,n);
    
max1=0;     %Value for minimum action
max2=42;    %Value for maximum action (42m/s = 
zeroone = linspace(max1,max2,27);
acttest = ones(27,n);
for k = 1:27
    acttest(k,:) = zeroone(k).*ones(1,n);
end

for j=1:2 %POR QUE TEM ESSE J AQUI?
    for k = 1:27 % find the Q value for all of the input actions
        if in == 13
            Q(k,:)=crt([acttest(k,:);actst(2,:);actst(3,:);actst(4,:);actst(5,:);actst(6,:);actst(7,:);actst(8,:);actst(9,:);actst(10,:);actst(11,:);actst(12,:);actst(13,:);actst(14,:)]);
        end
    end
    for k = 1:n
        [~, ind1(k)] = max(Q(:,k));
        Q(ind1,k) = -Inf;
        [~, ind2(k)] = max(Q(:,k));
        max1(k) = acttest(ind1(k),k);
        max2(k) = acttest(ind2(k),k);
        acttest(:,k)= linspace(max1(k),max2(k),27)'; 
    end
    
end

net=feedforwardnet([30 30]);                % Create the network with 2 hidden layers with 30 nodes each
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
fprintf('...(Create_Actor) Actor training\n')
net = train(net,Minibatch{3,1},max1);       % Train to create the array with correct positions of the weights

%% Evaluation plot

% x=linspace(0,length(action),length(action));
% actionnet=net(Minibatch{3,1});
% plot(x,action,x,cell2mat(actionnet));

end
function act = train_actor(Minibatch,crt,act)

actst = cell2mat(Minibatch{1,1});  % Take action and states vector 
n = length(actst);                 % Find the number of samples
in = (crt.numInputs - 1);          % Find the number of inputs 

%fprintf('...Calculating the best action(train actor)\n');

% actst = cell2mat(Minibatch{1,1});  % Take action and states vector 
% n = length(actst);                 % Find the number of samples
% in = (crt.numInputs - 1);          % Find the number of inputs 

Q=zeros(27,n);
action=zeros(1,n);
    
max1=0;     %Value for minimum action
max2=42;    %Value for maximum action       
zeroone = linspace(max1,max2,27);
acttest = ones(27,n);
for k = 1:27 %Total of desired actions, states and next states (without function and rewards)
    acttest(k,:) = zeroone(k).*ones(1,n);
end

for j=1:2 %POR QUE TEM ESSE J AQUI?
    for k = 1:27 % find the Q value for all of the input actions
        if in == 13 %(6 distaces, 6 velocities, vv, gp and break)
            Q(k,:)=crt([acttest(k,:);actst(2,:);actst(3,:);actst(4,:);actst(5,:);actst(6,:);actst(7,:);actst(8,:);actst(9,:);actst(10,:);actst(11,:);actst(12,:);actst(13,:);actst(14,:)]);            
        end
    end
    for k = 1:n
        [~, ind1(k)] = max(Q(:,k));
        Q(ind1,k) = -Inf;
        [~, ind2(k)] = max(Q(:,k));
        max1(k) = acttest(ind1(k),k);
        max2(k) = acttest(ind2(k),k);
        acttest(:,k)= linspace(max1(k),max2(k),27)'; 
    end
end

fprintf('...(Train_Actor) Actor training\n')
act= train(act,Minibatch{3,1},max1); % Train to create the array with correct positions of the weights

%% Evaluation plot

% x=linspace(0,length(action),length(action));
% actionnet=net(Minibatch{3,1});
% plot(x,action,x,cell2mat(actionnet));

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
net.trainParam.epochs = 1;               % Use just one pass through network as long this learnig is not important
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

fprintf('...Critic weights are reseted\n')
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

fprintf('...(Create_Critic) Critic training\n')
net = train(net,Minibatch{1,1},Minibatch{2,1}); % Train to create the array with correct positions of the weights
end
function crt = train_critic(Minibatch,crt)

fprintf('...(Train_Critic) Critic training\n')
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
if netin==13 %States cosidered (6 distaces, 6 velocities, vv, gp and break)
    state={rawbatch(2,:);rawbatch(3,:);rawbatch(4,:);rawbatch(5,:);rawbatch(6,:);rawbatch(7,:);rawbatch(8,:);rawbatch(9,:);rawbatch(10,:);rawbatch(11,:);rawbatch(12,:);rawbatch(13,:);rawbatch(14,:)};
    nextstate={rawbatch(17,:);rawbatch(18,:);rawbatch(19,:);rawbatch(20,:);rawbatch(21,:);rawbatch(22,:);rawbatch(23,:);rawbatch(24,:);rawbatch(25,:);rawbatch(26,:);rawbatch(27,:);rawbatch(28,:);rawbatch(29,:)};
    actionstate={rawbatch(1,:);rawbatch(2,:);rawbatch(3,:);rawbatch(4,:);rawbatch(5,:);rawbatch(6,:);rawbatch(7,:);rawbatch(8,:);rawbatch(9,:);rawbatch(10,:);rawbatch(11,:);rawbatch(12,:);rawbatch(13,:);rawbatch(14,:)};
    Qvalue= rawbatch(15,:);
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

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |
%|Des.  |dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ... 
%|VVL   |      |      |      |      |      |      |

%              |(:,8) ||(:,9)|(:,10)|(:,11)|(:,12)|(:,13)|(:,14)|
%          ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf|VVL   | ...
%              |      |      |      |      |      |      |      |

%|(:,15)|(:,16)|(:,17)|(:,18)|(:,19)|(:,20)|(:,21)|(:,22)|
%|TTC   |REWARD|dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ...
%|      |      |next  |next  |next  |next  |next  |next  |

%              |(:,23)|(:,24)|(:,25)|(:,26)|(:,27)|(:,28)|(:,29)|
%          ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf| VVL  |
%              |next  |next  |next  |next  |next  |next  |next  |
end
function [DATA,leng]=Simulation_treatment(safezone,TVL)

load('simul_inicial_treated.mat');
rawdata=stateaction.Data; 
%|(:,1)Des.VVL(:,2)dsx_ol(:,3)dsx_of(:,4)dsx_ll(:,5)dsx_lf(:,6)dsx_rl(:,7)dsx_rf
%|(:,8)dvx_ol(:,9)dvx_of(:,10)dvx_ll(:,11)dvx_lf(:,12)dvx_rl(:,13)dvx_rf(:,14)vvl

%% Maximum values 
%It finds the maximun values in the rawdata matrix to normalize it after

MaxDsx   = 0;  %Radar range (m)  ##150
MaxDvx   = 0; %Max. velocity (mph) ##50

[sizei,sizej]=size(rawdata);

%Normalizes the values between 0 and 1 to improve the learning performance
for j=1:sizej
    if and(j>=2,j<=7)%Columns of distance values
        MaxDsx=max(abs(rawdata(:,j))); %Save maximum range(m)
        if (MaxDsx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxDsx;
        end
    elseif and(j>=8,j<=14)%Columns of velocity values
        MaxDvx=max(abs(rawdata(:,j))); %Save maximum velocity(mph)
        if (MaxDvx==0)
            rawdata(:,j) = 0;
        else
            rawdata(:,j) = rawdata(:,j)./MaxDvx;
        end
    end
end

%% Data treatment
%Normalizes the values between 0 and 1 to improve the learning performance

% for i=1:sizej
%     if and(i>=3,i<=8)%Columns of distance values
%         rawdata(:,i) = rawdata(:,i)./MaxDsx;
%     elseif and(i>=9,i<=14)%Columns of velocity values
%         rawdata(:,i) = rawdata(:,i)./MaxDvx;
%     end
% end

leng = length(rawdata(:,1))-1;

rawdata=calculate_reward(rawdata,safezone,TVL);

%%                     rawdata matrix

%|(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |
%|Des.  |dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ... 
%|VVL   |      |      |      |      |      |      |

%       |(:,8) |(:,9) |(:,10)|(:,11)|(:,12)|(:,13)|(:,14)|
%   ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf| VVL  | ...
%       |      |      |      |      |      |      |      |

%       |(:,15)|(:,16)|(:,17)|(:,18)|(:,19)|(:,20)|
%   ... |dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ...
%       |next  |next  |next  |next  |next  |next  |

%       |(:,21)|(:,22)|(:,23)|(:,24)|(:,25)|(:,26)|(:,27)|(:,28)|(:,29)|
%   ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf| VVL  | TTC  |REWARD|
%       |next  |next  |next  |next  |next  |next  | next |      |      |

% Create the export data structure
DATA=rawdata(1:leng,[1:14 28 29 15:27]);
save ('DATA.mat','DATA');
fprintf('DATA Updated. First Rewards Assigned!\n')

%%                     DATA matrix

%       |(:,1) |(:,2) |(:,3) |(:,4) |(:,5) |(:,6) |(:,7) |
%       |Des.  |dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ... 
%       |VVL   |      |      |      |      |      |      |

%              |(:,8) |(:,9) |(:,10)|(:,11)|(:,12)|(:,13)|(:,14)|
%          ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf| VVL  | ...
%              |      |      |      |      |      |      |      |

%|(:,15)|(:,16)|(:,17)|(:,18)|(:,19)|(:,20)|(:,21)|(:,22)|
%| TTC  |REWARD|dsx_ol|dsx_of|dsx_ll|dsx_lf|dsx_rl|dsx_rf| ...
%|      |      |next  |next  |next  |next  |next  |next  |

%              |(:,23)|(:,24)|(:,25)|(:,26)|(:,27)|(:,28)|(:,29)|
%          ... |dvx_ol|dvx_of|dvx_ll|dvx_lf|dvx_rl|dvx_rf| VVL  |
%              |next  |next  |next  |next  |next  |next  | next |

end
function rawdata   = calculate_reward(rawdata,safezone,TVL)

leng = length(rawdata(:,1))-1;

%Copy the next states to the same line that the last state 
for i=1:leng
    for j=1:13
        if (i==leng)
            rawdata(i,(j+14))=rawdata(i,(j+1));
        else
            rawdata(i,(j+14))=rawdata(i+1,(j+1));
        end
    end
end
rawdata(:,28)=ones(); %future rewards (28)-->>(15)
rawdata(:,29)=ones(); %future TTC     (29)-->>(16)
[sizei,sizej]=size(rawdata);

%% #### Reward THIS IS THE BIGGEST CHALLENGE HERE!!! #### %%

%% TTC-based Reward ('Time To Collision' against leading car) and TTS-Based Reward ('Time To Stop' VUT)
RewardTTC=0;
RewardDelT=0;
for i=1:leng % Calculate TTC at the next step and save in rawdata(:,33)
    TTC=time_to_collision(rawdata(i,15),rawdata(i,21),safezone); %Generate TTC for lane O
    TTS=(0.323.*rawdata(i,27)+1.999); %Time for total stop %Fitted for currently car
    DelT=TTC./TTS; %Time to collision compared to time for total stop
    if (TTC>0) %Positive Time to Collision
        if and(rawdata(i,15)>safezone,rawdata(i,21)>0)      %No collision/Leading vehicle moving away//Best situation
            RewardTTC=0.5;
        elseif and(rawdata(i,15)<safezone,rawdata(i,21)<0)  %Collision/Leading vehicle approaching//Worst situation
            RewardTTC=0;
        end       
    elseif (TTC<=0) %Negative Time to Collision
        if and(rawdata(i,15)>safezone,rawdata(i,21)<0)      %No collision/Leading vehicle approaching
            RewardTTC=0.4;
        elseif and(rawdata(i,15)<safezone,rawdata(i,21)>0)  %Collision/Leading vehicle moving away
            RewardTTC=0;
        end
    elseif (DelT>1) 
        RewardDelT=1; %Best situation, when TTC is bigger than TTS; 
    else
        RewardDetT=0; %For situations where TTC is shorter and there will be collision
    rawdata(i,29)=TTC;
    rawdata(i,28)=RewardTTC+RewardDetT;    
    end
end
%% VVL-Based Reward (Real and Desired Velocities) 
RewardDelVVL=0;
for i= 1:leng
    DelVVL=((minus(rawdata(i,27),TVL))./TVL);
    if DelVVL>=1.1                      %Real velocity over 110% os desired one
        RewardDelVVL=0.6;
    elseif and(DelVVL<1.1,DelVVL>=1)    %Real velocity between 100-110%
        RewardDelVVL=1;
    elseif and(DelVVL<1,DelVVL>=0.8)    %Real velocity between 80-100%
        RewardDelVVL=0.6;
    elseif and(DelVVL<0.8,DelVVL>=0.6)  %Real velocity between 60-80%
        RewardDelVVL=0.4;
    elseif and(DelVVL<0.6,DelVVL>=0.4)  %Real velocity between 40-60%
        RewardDelVVL=0.2;
    elseif DelVVL<0.4                   %Real velocity under 40%
        RewardDelVVL=0;
    rawdata(i,28)=rawdata(i,28)+RewardDelVVL; %SUM with last reward
    end
end
%% TTS-Based Reward ('Time To Stop' VUT)
RewardDelT=0;
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


function plot_convergence (Qa,Qb,actaa,actbb,hv,acterror);

figure (1);
subplot(2,2,1)
plot(Qb,'b');
hold on
subplot(2,2,1)
plot(Qa,'r');
legend('Updated','Calculated');
ylabel('Q-value');
xlabel('Amount of data');
hold off

subplot(2,2,2)
plot(actbb,'b');
hold on
subplot(2,2,2)
plot(actaa ,'r');
legend('Updated','Calculated');
ylabel('Action');
xlabel('Amount of data');
hold off

subplot(2,2,[3,4]);
plot(hv,acterror,'-o');  %Plot Q error convergence
xlabel('Training cycle');
ylabel('Action error');

end
