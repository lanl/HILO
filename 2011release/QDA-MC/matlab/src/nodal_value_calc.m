function [nv] = nodal_value_calc(v,data)

%INITIALIZE SIMULATION PARAMETERS
nx = data.nx;

%CALCULATE THE NODAL VALUES
nv = zeros(nx,1);
for i = 1:nx
    nv(i) = (v(i)+v(i+1))/2;
end