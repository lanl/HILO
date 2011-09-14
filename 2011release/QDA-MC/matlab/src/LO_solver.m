function [phi_LO] = LO_solver(data,phi_HO,J_HO,E_HO,E_HO_n,ic)
%INITIALIZE SIMULATION PARAMETERS
%ic = flag for initial condition or not.
nx = data.nx;
dx = data.dx;
sig_t = data.sig_t;
sig_s = data.sig_s;
alp = 1/(dx*dx*sig_t);
beta = sig_t - sig_s;
gam = sig_s/(dx*sig_t*sig_t);
D = data.D;
Q0_LO = data.Q0_LO;
b(1:nx,1) = Q0_LO;
A = sparse(nx,nx);%initialize the sparse matrix system for LO system
if ic == 0 %not initial condition
    %DECLARE THE BOUNDARY VALUES FOR THE LO SYSTEM
    phi_l = E_HO_n(1)*(E_HO(1) - J_HO(1)/phi_HO(1)/2/gam)^(-1);
    phi_r = E_HO_n(nx)*(E_HO(nx+1) + J_HO(nx+1)/phi_HO(nx+1)/2/gam)^(-1);
    for i = 1:nx
        if i == 1 %left boundary cell
            A(i,i) = beta + ...
                     gam*E_HO_n(1)/dx - ...
                     phi_l*J_HO(1)/phi_HO(1)/dx;
            A(i,i+1) = -gam*E_HO_n(2)/dx;
        elseif i == nx %right boundary cell
            A(i,i) = beta + ...
                     gam*E_HO_n(nx)/dx + ...
                     phi_r*J_HO(nx+1)/phi_HO(nx+1)/dx;
            A(i,i-1) = -gam*E_HO_n(nx-1)/dx;
        else %internal cell
            A(i,i) = 2*alp*E_HO_n(i) + beta;
            A(i,i+1) = -alp*E_HO_n(i+1);
            A(i,i-1) = -alp*E_HO_n(i-1);
        end
    end
else%if initial condition
    %DECLARE THE BOUNDARY VALUES FOR THE LO SYSTEM
    phi_l = (4*D/dx/(sig_s/sig_t + 4*D/dx));
    phi_r = (4*D/dx/(sig_s/sig_t + 4*D/dx));
    for i = 1:nx
        if i == 1 %left boundary cell
            A(i,i) = beta + ...
                     2*alp*E_HO(1)*phi_l - ...
                     3*alp*E_HO_n(1);
            A(i,i+1) = alp*E_HO_n(2);
        elseif i == nx %right boundary cell
            A(i,i) = beta + ...
                     2*alp*E_HO(nx+1)*phi_r - ...
                     3*alp*E_HO_n(nx);
            A(i,i-1) = alp*E_HO_n(nx-1);
        else %internal cell
            A(i,i) = 2*alp*E_HO_n(i) + beta;
            A(i,i+1) = -alp*E_HO_n(i+1);
            A(i,i-1) = -alp*E_HO_n(i-1);
        end
    end    
end
phi_LO = A\b; % Need Tri-diagonal linear solver
% Thompson algorithm? Sparse Matrix Solvers?
