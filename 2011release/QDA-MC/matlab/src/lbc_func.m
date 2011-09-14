function [lbc_NP,lbc_mu,lbc_wp] = lbc_func(data)

%INITIALIZATION OF SIMULATION PARAMETERS
lbc_flag = data.lbc_flag;
NP = data.NP;
Jp_l = data.Jp_l;
%SET THE LEFT BOUNDARY CONDITION TYPE
if lbc_flag == 0 %isotropic source
    lbc_NP = round(Jp_l*NP);
    lbc_mu = rand(lbc_NP,1);
    lbc_wp(1:NP,1) = 1;
elseif lbc_flag == 1 %beam source
    lbc_NP = round(Jp_l*NP);
    lbc_mu(1:lbc_NP,1) = 1;%forward peak beam source
    lbc_wp(1:NP,1) = 1;
else %vacuum boundary
    lbc_NP = 0;
    lbc_mu = 0;
    lbc_wp = 0;
end