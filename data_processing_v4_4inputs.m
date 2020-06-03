clear; close all; clc;

%% Loading sensor signal from last simulation

load simul_inicial.mat
disp('simul_inicial.mat opened');

RAW=data.data;
timevector=data.Time;
clear data

DATA = [RAW(:,4), RAW(:,1), RAW(:,2), RAW(:,3), RAW(:,4)];
%% at the moment, desired is equal to current actions!!!

stateaction=timeseries(DATA,timevector);
stateaction.Name='';
save simul_inicial_treated.mat stateaction;
disp('simul_inicial_treated.mat saved');

%% Clear all residual variables -to improve next step's performance

clear DATA RAW timevector stateaction