function sim_param_print(data)

%INITIALIZATION OF SIMULATION PARAMETERS
lx = data.lx;
dx = data.dx;
nx = data.nx;
NP = data.NP;
sig_t = data.sig_t;
sig_s = data.sig_s;
prob_s = data.prob_s;

fprintf('***********************************************************');
fprintf('\n');
fprintf('******THE SIMULATION PARAMETERS ARE PRINTER OUT BELOW******');
fprintf('\n');
fprintf('The system length is, lx = %g', lx);
fprintf('\n');
fprintf('The cell width is, dx = %g', dx);
fprintf('\n');
fprintf('The number of cell is, nx = %g', nx);
fprintf('\n');
fprintf('The reference number of particle is, NP = %g', NP);
fprintf('\n');
fprintf('The total cross section is, sig_t = %g', sig_t);
fprintf('\n');
fprintf('The scattering cross section is, sig_s = %g', sig_s);
fprintf('\n');
fprintf('The scattering probability is, prob_s = %g', prob_s);
fprintf('\n');
fprintf('***********************************************************');
fprintf('\n');