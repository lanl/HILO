function rel_diff = rel_diff_calc(data,phi,phi_old)
%INITIALIZE SIMULATION PARAMETERS
nx = data.nx;
rel_diff = zeros(nx,1);

for i = 1:nx
    rel_diff(i) = abs((phi(i)-phi_old(i))/phi(i));
end

rel_diff = max(rel_diff);
fprintf('The relative difference between the scalar flux is, %g',rel_diff);
fprintf('\n');