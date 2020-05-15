clear; close all; clc;

%% Loading sensor signal from last simulation

load simul_inicial.mat
traffic_nObjs = data.traffic_nObjs;                                
traffic_dsy = data.traffic_dsy;
traffic_dsx = data.traffic_dsx;
traffic_dvx = data.traffic_dvx;
traffic_dvy = data.traffic_dvy;
traffic_bearing = data.traffic_bearing;
traffic_length = data.traffic_length;
traffic_width = data.traffic_width;
ego_vx = data.ego_vx;
ego_gp = data.ego_gp;
ego_brake = data.ego_brake;
timevector=ego_gp.Time;
clear data

%% Addressing Distances (dsx/dsy), Velocities (dvx/dvy), Gas pedal (gp) and Brake pedal (break)

lanewidth=1.75; %3.5m (1.75 for the right and 1.75 for the right)

sizei=traffic_dsx.length;
sizej=traffic_nObjs.data(1,1);
for i=1:sizei
    dsx(i,:)=traffic_dsx.data(i,:);                 %Position from sensor in x-coordinate
    dsy(i,:)=traffic_dsy.data(i,:);                 %Position from sensor in y-coordinate
    dvx(i,:)=traffic_dvx.data(i,:);                 %Relative Velocity between sensor and traffic object in x-coordinate
    dvy(i,:)=traffic_dvy.data(i,:);                 %Relative Velocity between sensor and traffic object in y-coordinate
    vvl(i,:)=ego_vx.data(i,:);                      %Absolute Velocity of the ego vehicle
    gp(i,:)=ego_gp.data(i,:);                       %Gas pedal position of the ego vehicle
    brake(i,:)=ego_brake.data(i,:);                 %Brake position of the ego vehicle
    
    for j=1:sizej  %%Optimize!!! 1.75 (width of each column sensor scan
        if (dsy(i,j)<=lanewidth) & (dsy(i,j)>=-lanewidth) & (dsy(i,j)~=0)     %Object is in Lane 0 
            disp('Lane 0') 
            if (i~=1)
                dsxpos=find(dsx(i,:)>0,1,'first');
                if (dsxpos>0)
                    disp('Lane 0 - Positive Improved')
                    dsx_ol(i,1)=dsx(i,dsxpos(1,1));
                    dvx_ol(i,1)=dvx(i,dsxpos(1,1));
                else
                    disp('Lane 0 - Positive No Improved')
                    dsx_ol(i,1)=dsx_ol(i-1,1);
                    dvx_ol(i,1)=dvx_ol(i-1,1);
                end
                dsxneg=find(dsx(i,:)<0,1,'last');
                if (dsxneg>0)
                    disp('Lane 0 - Negative Improved')
                    dsx_of(i,1)=dsx(i,dsxneg(1,1));
                    dvx_of(i,1)=dvx(i,dsxneg(1,1));
                else
                    disp('Lane 0 - Negative No Improved')
                    dsx_of(i,1)=dsx_of(i-1,1);
                    dvx_of(i,1)=dvx-of(i-1,1);
                end
                lane="0";
            end
        elseif (dsy(i,j)>lanewidth)                       %Object is in Lane L
            disp('Lane L') 
            if (i~=1)
                dsxpos=find(dsx(i,:)>0,1,'first');
                if (dsxpos>0)
                    disp('Lane L - Positive Improved')
                    dsx_ll(i,1)=dsx(i,dsxpos(1,1));
                    dvx_ll(i,1)=dvx(i,dsxpos(1,1));
                else
                    disp('Lane L - Positive No Improved')
                    dsx_ll(i,1)=dsx_ll(i-1,1);
                    dvx_ll(i,1)=dvx_ll(i-1,1);
                end
                dsxneg=find(dsx(i,:)<0,1,'last');
                if (dsxneg>0)
                    disp('Lane L - Negative Improved')
                    dsx_lf(i,1)=dsx(i,dsxneg(1,1));
                    dvx_lf(i,1)=dvx(i,dsxneg(1,1));
                else
                    disp('Lane L - Negative No Improved')
                    dsx_lf(i,1)=dsx_lf(i-1,1);
                    dvx_lf(i,1)=dvx_lf(i-1,1);
                end
                lane="L";
            end
        elseif (dsy(i,j)<-lanewidth)                      %Object is in Lane R
            disp('Lane R') 
            if (i~=1)
                dsxpos=find(dsx(i,:)>0,1,'first');
                if (dsxpos>0)
                    disp('Lane R - Positive Improved')
                    dsx_rl(i,1)=dsx(i,dsxpos(1,1));
                    dvx_rl(i,1)=dvx(i,dsxpos(1,1));
                else
                    disp('Lane R - Positive No Improved')
                    dsx_rl(i,1)=dsx_rl(i-1,1);
                    dvx_rl(i,1)=dvx_rl(i-1,1);                    
                end
                dsxneg=find(dsx(i,:)<0,1,'last');
                if (dsxneg>0)
                    disp('Lane R - Negative Improved')
                    dsx_rf(i,1)=dsx(i,dsxneg(1,1));
                    dvx_rf(i,1)=dvx(i,dsxneg(1,1));
                else
                    disp('Lane R - Negative No Improved')
                    dsx_rf(i,1)=dsx_rf(i-1,1);
                    dvx_rf(i,1)=dvx_rf(i-1,1);
                end
                lane="R";
            end
        elseif (dsy(i,j)==0)                          %Empty value
            if (i>1)&(lane=="0");
                disp('Keeping Last Value 0')
                dsx_ll(i,1)=dsx_ll(i-1,1);
                dsx_lf(i,1)=dsx_lf(i-1,1);
                dsx_rl(i,1)=dsx_rl(i-1,1);
                dsx_rf(i,1)=dsx_rf(i-1,1);
                dvx_ll(i,1)=dvx_ll(i-1,1);
                dvx_lf(i,1)=dvx_lf(i-1,1);
                dvx_rl(i,1)=dvx_rl(i-1,1);
                dvx_rf(i,1)=dvx_rf(i-1,1);
            elseif (i>1)&(lane=="L");
                disp('Keeping Last Value L')
                dsx_ol(i,1)=dsx_ol(i-1,1);
                dsx_of(i,1)=dsx_of(i-1,1);
                dsx_rl(i,1)=dsx_rl(i-1,1);
                dsx_rf(i,1)=dsx_rf(i-1,1);
                dvx_ol(i,1)=dvx_ol(i-1,1);
                dvx_of(i,1)=dvx_of(i-1,1);
                dvx_rl(i,1)=dvx_rl(i-1,1);
                dvx_rf(i,1)=dvx_rf(i-1,1);
            elseif (i>1)&(lane=="R");
                disp('Keeping Last Value R')
                dsx_ol(i,1)=dsx_ol(i-1,1);
                dsx_of(i,1)=dsx_of(i-1,1);
                dsx_ll(i,1)=dsx_ll(i-1,1);
                dsx_lf(i,1)=dsx_lf(i-1,1);
                dvx_ol(i,1)=dvx_ol(i-1,1);
                dvx_of(i,1)=dvx_of(i-1,1);
                dvx_ll(i,1)=dvx_ll(i-1,1);
                dvx_lf(i,1)=dvx_lf(i-1,1);
            elseif (i==1)&(j==1)
                disp('Initial Condition With No Objects')
                dsx_ol(1,1)=150; 
                dsx_of(1,1)=-150; 
                dsx_ll(1,1)=150; 
                dsx_lf(1,1)=-150;
                dsx_rl(1,1)=150; 
                dsx_rf(1,1)=-150;
                dvx_ol(1,1)=0; 
                dvx_of(1,1)=0; 
                dvx_ll(1,1)=0; 
                dvx_lf(1,1)=0;
                dvx_rl(1,1)=0; 
                dvx_rf(1,1)=0;
            end  
        end
%         if(j==14)                     %for code debugging
%             break
%         end
    end
%     if(i==3)                          %for code debugging
%         break
%     end
end

% A=ones(1,sizei);
% DATA = [1:30];
DATA = [vvl, dsx_ol, dsx_of, dsx_ll, dsx_lf, dsx_rl, dsx_rf, dvx_ol, dvx_of, dvx_ll, dvx_lf, dvx_rl, dvx_rf, vvl];
%% at the moment, desired is equal to current actions!!!
% We need to implement the random input.

% for i=1:sizei
%     for j=1:14
%         if (i==sizei)
%             DATA(i,(j+16))=DATA(i,(j+2));
%         else
%             DATA(i,(j+16))=DATA(i+1,(j+2));
%         end
%     end
% end

stateaction=timeseries(DATA,timevector);
stateaction.Name='';
save simul_inicial_treated.mat stateaction;
disp('DATA Treated and Generated');

%% Example for Near Object data saving (used above)
% 
% for i=95:95 
%     dsxpos=find(dsx(i,:)>0,1,'first')
%     dsx_rl=dsx(i,dsxpos(1,1))
%     dvx_rl=dvx(i,dsxpos(1,1)) 
% end
% for i=95:95
%     dsxneg=find(dsx(i,:)<0,1,'last')
%     dsx_rf=dsx(i,dsxneg(1,1))
%     dvx_rf=dvx(i,dsxneg(1,1)) 
% end

%% Clear all residual variables -to improve next step's performance

clear dsx dsy dvx dvy dsx_ol dsx_of dsx_ll dsx_lf dsx_rl dsx_rf dvx_ol dvx_of dvx_ll dvx_lf dvx_rl dvx_rf vvl gp brake dsxpos dsxneg i j lane size ego_vx traffic_length traffic_width traffic_nObjs traffic_bearing traffic_dsx traffic_dsy traffic_dvx traffic_dvy sizei sizej ego_v ego_gp ego_brake A DATA timevector lanewidth
