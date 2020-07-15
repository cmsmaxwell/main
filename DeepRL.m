function output = DeepRL(dr_ol,vr_ol,v_vut,pedal,simCount,chosen); 
load act.mat
output = actb([dr_ol;vr_ol;v_vut;pedal]);
end