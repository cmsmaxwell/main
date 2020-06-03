function output = DeepRL(dr_ol,vr_ol,v_vut,pedal,simCount,chosen); 
load act.mat
output = actb([dr_ol;vr_ol;v_vut;pedal]);
decision=chosen*rand();
if decision >= 0.9
    output=round(rand(),2);
else
    output=round(output,2);
end

end