function [average,std] = mean_std_calc(value,value2,samp_cnt,data)
%INITIALIZATION OF SIMULATION PARAMETERS
NP = data.NP;
value = value*NP;
average = value./samp_cnt;
std = sqrt(((value2/samp_cnt) - (average.^2))/(samp_cnt - 1));