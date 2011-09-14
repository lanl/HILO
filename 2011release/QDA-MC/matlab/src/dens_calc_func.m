function pn = dens_calc_func(x,xpos_face)
%INITIALIZE SIMULATION PARAMETERS
nx = length(xpos_face) - 1;
N = length(x);
pn = zeros(nx,1);
% START THE PARTICLE CELL RESIDENCY TAGGING
i = 0;
j = 0;
while j < N%loop through all particles
    i = i + 1;%loop through all cell
    flag = 0;%flag to kick the user out of the below while loop
    while flag == 0
        j = j + 1;%next particle
        if (x(j) >= xpos_face(i)) && (x(j) <= xpos_face(i+1))
            pn(i) = pn(i) + 1;%store the initial cell location for the j'th particle
            if j == N
                flag = 1;
            end
        else
            flag = 1;j = j - 1;
        end
    end
end
