function [x,mu] = placement_func(data,n)

%INITIALIZE THE SIMULATION PARAMETERS%
dx = data.dx;
nx = data.nx;
NP = data.NP;
xpos_node = data.xpos_node;
tally = 0;%tally for vector size extension
NP_fluc = round(NP*n);%This is the number of particle per cell
vec_act = 0;
if sum(NP_fluc) == 0 %no particle exists in the system
    x(1,1) = 0;
    mu(1,1) = 0;
else %if there's even one particle in the system
    for i = 1:nx
        mux = 0;
        xx = 0;
        ddx = dx/NP_fluc(i);
        if NP_fluc(i) == 0
        else
            vec_act = vec_act + 1;
            xx(1:NP_fluc(i),1) = 0;
            mux(1:NP_fluc(i),1) = 0;
            for k = 1:NP_fluc(i)
                xx(k) = xpos_node(i) - dx/2 + ddx*(k-1/2);
                mux(k) = 2*rand - 1;%random isotropic assignment of cosine theta
            end
        end
        if vec_act == 1
            x = xx;
            mu =mux;
        else
            x = [x;xx];
            mu = [mu;mux];
        end
    end
end