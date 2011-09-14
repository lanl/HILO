function collision_tester(N,lx,nx)
Sig_t = 1;
N0 = N;
dx = lx/nx;%cell width for tallying
x = zeros(N,1);
% mu = rand(N,1);
mu(1:N,1) = 1;
%CALCULATE THE CELL CENTER AND FACE
xpos_face(1:nx+1,1) = 0;
xpos_node(1:nx,1) = 0;  
for i = 1:nx+1
    xpos_face(i) = (i-1)*dx;
end
for i = 1:nx
    if i == 1
        xpos_node(i) = dx/2;
    elseif i == nx
        xpos_node(i) = (i*dx) - dx/2;
    else
        xpos_node(i) = (i-1)*dx + dx/2;
    end
end
%STREAM THE PARTICLES
for i = 1:N
    x(i) = x(i) + mu(i)*(-log(rand)/Sig_t);
    if x(i) >= lx || x(i) <= 0 %check if particle is outside of system
        x(i) = sqrt(-1);
    end
    a = rand;%random number to see if the 
end
x(x == sqrt(-1)) = [];
N = length(x);
x = sort(x);%SORT THE PARTICLES SO THEY ARE IN ORDER FOR EASY TALLYING

n = zeros(nx);
%ACTUAL MOMENT CALCULATION
j = 0;
i = 0;
while j < N
    i = i + 1;
    flag = 0;
    while flag == 0
        j = j + 1;%next particle
        if (x(j) >= xpos_face(i)) && (x(j) < xpos_face(i+1))
            n(i) = n(i) + 1;
            if j == length(x)
                flag = 1;
            end
        else
            flag = 1;j = j - 1;
        end
    end
end
n = n/n(1);
for i = 1:nx
    P(i) = exp(-Sig_t*xpos_node(i));%Analytical Solution
end
plot(xpos_node,n);hold on;plot(xpos_node,P,'r');hold off;
