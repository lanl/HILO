function n = init_cond(data)

%INITIALIZATION OF SIMULATION PARAMETERS
dx = data.dx;
nx = data.nx;
n0 = data.n0;
%CALCULATION OF THE GRID CENTER MOMENT QUANTITIES
n = zeros(nx,1);
for i = 1:nx
    n(i) = n0;
end