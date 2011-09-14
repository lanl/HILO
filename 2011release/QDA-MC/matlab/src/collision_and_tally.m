function [phi,J,E,phi_HO_S2,phi_n,J_n,E_n,phi_HO_S2_n] = collision_and_tally(data,Qbar,phi_HO_S2,phi_HO_S2_n)
%INITIALIZE SIMULATION PARAMETERS
nx = data.nx;
xpos_node = data.xpos_node;
dx = data.dx;
NP = data.NP;
NPc = data.NPc;
QNP(1:nx,1) = NPc;
% QNP = round(Qbar*NP*dx);
lbc_flag = data.lbc_flag;
%FORM THE BOUNDARY PARTICLES FROM THE LEFT SURFACE
[lbc_NP,lbc_mu] = lbc_func(data);
%STREAM THE PARTICLES, COLLIDE AND CALCULATE MOMENT ALL SIMULTANEOUSLY
%CALCULATE THE FINAL LOCATION OF THE PARTICLE
phi = zeros(nx+1,1);
J = zeros(nx+1,1);%current is a face quantity and will be converted to cell averaged quantity later on
E = zeros(nx+1,1);
phi_n = zeros(nx,1);
J_n = zeros(nx,1);%current is a face quantity and will be converted to cell averaged quantity later on
E_n = zeros(nx,1);
%BRING IN NEUTRONS FROM LEFT BOUNDARY
 
%THE SCATTERING AND INTERNAL SOURCE TERM
tal = 0;
if sum(QNP) ~= 0
    xx = zeros(sum(QNP),1);
    mux = zeros(sum(QNP),1);
    wx = zeros(sum(QNP),1);%weight per particle
    for i = 1:nx
        if QNP(i) ~= 0
            if i == 1
                for j = 1:QNP(i)
                    xx(j) = xpos_node(i) + (-0.5 + rand)*dx;
                    mux(j) = -1+2*rand;
                    wx(j) = abs(Qbar(i)*NP*dx/NPc);%weight of each particles
                end
            else
                for j = 1:QNP(i)
                    xx(j + tal) = xpos_node(i) + (-0.5 + rand)*dx;
                    mux(j + tal) = -1+2*rand;
                    wx(j + tal) = abs(Qbar(i)*NP*dx/NPc);%weight of each particles
                end
            end
        end
        tal = tal + QNP(i);
    end
    x0 = xx;
    mu0 = mux;
    [xx,mux,xf,cell0,cellf] = xmc0f_calc(data,xx,mux,x0);
    %SORT THE PARTICLES FOR EASY CELL ACCESSING
    [phi,J,E,phi_HO_S2] = face_moment_calc(x0,mu0,wx,xf,cell0,cellf,phi,J,E,phi_HO_S2,data);
    [phi_n,J_n,E_n,phi_HO_S2_n] = node_moment_calc(x0,mu0,wx,xf,cell0,cellf,phi_n,J_n,E_n,phi_HO_S2_n,data);
end
%SCALING THE MOMENTS FOR FACE VALUES
for i = 1:nx+1
    if phi(i) ~= 0
        E(i) = E(i)/phi(i);
    else
        E(i) = 1/3;
    end
end
phi = phi/NP;%scale and re-normalize the phi after all interaction
J = J/NP;%scale and re-normalize the J after all interaction
%SCALING THE MOMENTS FOR NODE VALUES
for i = 1:nx
    if phi_n(i) ~= 0
        E_n(i) = E_n(i)/phi_n(i);
    else
        E_n(i) = 1/3;
    end
end
phi_n = phi_n/dx/NP;
J_n = J_n/dx/NP;
