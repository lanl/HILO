function [xx,mux,xf,cell0,cellf] = xmc0f_calc(data,xx,mux,x0)
%INITIALIZE SIMULATION PARAMETERS
prob_s = data.prob_s;
lx = data.lx;
sig_t = data.sig_t;
% xpos_node = data.xpos_node;
xpos_face = data.xpos_face;
N = length(xx);
cell0 = part_cell_pos_func(x0,xpos_face);%DETERMINE WHICH CELL THE PARTICLES INITIALLY RESIDED IN
xf = zeros(N,1);
for i = 1:N
    fx = (-log(rand)/sig_t);
    xx(i) = xx(i) + mux(i)*fx;
    xf(i) = xx(i);%this is the final particle location
    if xx(i) >= lx 
        xx(i) = sqrt(-1);
        xf(i) = lx;
        mux(i) = sqrt(-1);
    elseif xx(i) <= 0 %check if particle is outside of system
        xx(i) = sqrt(-1);
        xf(i) = 0;
        mux(i) = sqrt(-1);
    else%if particle is inside the system see what collision it undergoes
        a = rand;%random number to investigate the type of collision
        if a <= prob_s %scattering happens
            mux(i) = -1 + 2*rand;
        else %absorption occurs
            xx(i) = sqrt(-1);
            mux(i) = sqrt(-1);
        end
    end
end
%SORT THE PARTICLES FOR EASY CELL ACCESSING
[xf,IX] = sort(xf);%sort the xf for easy cell accessing
cellf = part_cell_pos_func(xf,xpos_face);%DETERMINE WHICH CELL THE PARTICLES LANDED ON AFTER THE FLIGHT
xf = reverse_sort(xf,IX);
cellf = reverse_sort(cellf,IX);
% pn = dens_calc_func(xx,xpos_face);
% figure(2000);plot(xpos_node,pn,'g');xlabel('x');ylabel('n');pause