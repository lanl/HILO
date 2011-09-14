function [x,mu] = collision(data,x,mu)
%INITIALIZE SIMULATION PARAMETERS
prob_s = data.prob_s;
sig_t = data.sig_t;
sig_s = data.sig_s;
xpos_face = data.xpos_face;
nx = data.nx;
xpos_node = data.xpos_node;
NP = data.NP;
Q0 = data.Q0;
QNP = round(Q0*NP);
N = length(x);
lx = data.lx;
lbc_flag = data.lbc_flag;
lbc_NP = data.lbc_NP;
lbc_mu = data.lbc_mu;
%STREAM THE PARTICLES, COLLIDE AND CALCULATE MOMENT ALL SIMULTANEOUSLY
for i = 1:N %PAUL This could be a standalone kernel, with slight modification
    fx = (-log(1-rand)/sig_t);
    x(i) = x(i) + mu(i)*fx;
    if x(i) >= lx || x(i) <= 0 %check if particle is outside of system
        x(i) = sqrt(-1);
        mu(i) = sqrt(-1);
    else%if particle is inside the system see what collision it undergoes
        a = rand;%random number to investigate the type of collision
        if a >= prob_s %scattering happens
            mu(i) = 2*rand - 1;%randomly select the new direction
        else %absorption occurs
            x(i) = sqrt(-1);
            mu(i) = sqrt(-1);
        end
    end
end
%BRING IN NEUTRONS FROM LEFT BOUNDARY
if lbc_flag == 2%if vacuum, nothing happens
    xx = sqrt(-1);
    mux = sqrt(-1);
else
    xx = zeros(lbc_NP,1);
    mux = zeros(lbc_NP,1);
    for i = 1:lbc_NP %PAUL This could be a standalone kernel, with slight modification
        fx = (-log(1-rand)/sig_t);
        xx(i) = lbc_mu(i)*fx;
        if xx(i) >= lx || xx(i) <= 0 %check if particle is outside of system
            xx(i) = sqrt(-1);
            mux(i) = sqrt(-1);
        else%if particle is inside the system see what collision it undergoes
            a = rand;%random number to investigate the type of collision
            if a <= prob_s %scattering happens
                mux(i) = -1 + 2*rand;%randomly select the new direction
            else %absorption occurs
                xx(i) = sqrt(-1);
                mux(i) = sqrt(-1);
            end
        end
    end
end
x = [x;xx];
mu = [mu;mux];
%PRODUCE NEUTRONS FROM SOURCE TERM
if sum(QNP) ~= 0
    xx = zeros(sum(QNP),1);
    mux = zeros(sum(QNP),1);
    for i = 1:nx
        if QNP(i) ~= 0
            for j = 1:QNP(i)
                if i == 1
                    xx(j) = xpos_node(i);
                    mux(j) = -1+2*rand;
                else
                    xx(j + sum(QNP(1:i-1))) = xpos_node(i);
                    mux(j + sum(QNP(1:i-1))) = -1+2*rand;
                end
            end
        else
        end
    end
else
    xx = sqrt(-1);
    mux = sqrt(-1);
end
x = [x;xx];
mu = [mu;mux];
x(x == sqrt(-1)) = [];
mu(mu == sqrt(-1)) = [];
x = xmu_sort(x,mu);%SORT THE PARTICLES SO THEY ARE IN ORDER FOR EASY TALLYING
