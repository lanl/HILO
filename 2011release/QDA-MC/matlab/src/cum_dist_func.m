function [xx,phi_LO_f] = cum_dist_func(data,phi_LO,phi_HO,J_HO,E_HO,ic)
%INITIALIZE SIMULATION PARAMETERS
xpos_face = data.xpos_face;
dx = data.dx;
NP = data.NP;
NP_tot = data.NP_tot;
nx = data.nx;
sig_t = data.sig_t;
D = data.D;
gam = 1/(dx*sig_t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATE THE FACE CENTERED VALUES FOR THE FLUX
phi_LO_f = zeros(nx+1,1);
[E_HO_n] = nodal_value_calc(E_HO,data);
if ic == 0 %not initial condition
    phi_l = E_HO_n(1)*(E_HO(1) - J_HO(1)/phi_HO(1)/2/gam)^(-1);
    phi_r = E_HO_n(nx)*(E_HO(nx+1) + J_HO(nx+1)/phi_HO(nx+1)/2/gam)^(-1);
    for i = 1:nx+1
        if i == 1 %left boundary cell
            phi_LO_f(i) = phi_LO(i)/phi_l;
        elseif i == nx+1 %right boundary cell
            phi_LO_f(i) = phi_LO(i-1)/phi_r;
        else%linear interpolation of the nodal value is given to face
            phi_LO_f(i) = (phi_LO(i)+phi_LO(i-1))/2;
        end
    end
else %initial condition 
    for i = 1:nx+1
        if i == 1 %left boundary cell
            phi_LO_f(i) = phi_LO(i)/(2*D/dx/(1+2*D/dx));
        elseif i == nx+1 %right boundary cell
            phi_LO_f(i) = phi_LO(i-1)/(2*D/dx/(1+2*D/dx));            
        else%linear interpolation of the nodal value is given to face
            phi_LO_f(i) = (phi_LO(i)+phi_LO(i-1))/2;
        end
    end
end
% figure(1023);plot(xpos_face,phi_LO_f,'r');xlabel('x');ylabel('\phi^{LO}');title('\phi^{LO}');pause
%CUMULATIVE DISTRIBUTION FUNCTION CALCULATION
sum = 0;
sum_f = zeros(nx+1,1);
for i = 1:nx+1
    if i == 1
        sum = 0;
        sum_f(i) = sum;
    else
        sum = sum + phi_LO_f(i);
        sum_f(i) = sum;
    end
end
sum_f = sum_f/sum;
% figure(1021);plot(xpos_face,sum_f,'r');xlabel('x');ylabel('CDF');title('C
% DF');pause
%ADJUST THE TOTAL WEIGHT FOR THE PHI_LO with the initial weight = 1.0
xx_rand = rand(NP_tot,1);
xx_rand = sort(xx_rand);%sort the random numbers for easy accessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%CALCULATE THE PARTICLES VELOCITY BASED ON THE CDF%%%%%%%%%%%%%%%%
xx = zeros(NP_tot,1);
i = 0;
j = 0;
while j < NP_tot
    i = i + 1;%loop through all cell face and CDF mapping
    slope = (xpos_face(i+1)-xpos_face(i))/(sum_f(i+1)-sum_f(i));
    flag = 0;
    while flag == 0
        j = j + 1;%next particle
        if (xx_rand(j) >= sum_f(i)) && (xx_rand(j) <= sum_f(i+1))
            xx(j) = xpos_face(i) + slope*(xx_rand(j) - sum_f(i));
            if j == NP_tot
                flag = 1;
            end
        else
            flag = 1;j = j - 1;
        end
    end
end
xx = sort(xx);