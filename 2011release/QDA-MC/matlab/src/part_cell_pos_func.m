function cell = part_cell_pos_func(x,xpos_face)
%INITIALIZE SIMULATION PARAMETERS
N = length(x);
cell = zeros(N,1);
%START THE PARTICLE CELL RESIDENCY TAGGING
i = 0;
j = 0;
while j < N%loop through all particles
    i = i + 1;%loop through all cell
    flag = 0;%flag to kick the user out of the below while loop
    while flag == 0
        j = j + 1;%next particle
        if (x(j) >= xpos_face(i)) && (x(j) <= xpos_face(i+1))
            cell(j) = i;%store the initial cell location for the j'th particle
            if j == N
                flag = 1;
            end
        else
            flag = 1;j = j - 1;
        end
    end
end