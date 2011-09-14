function cell_pos_tal(data,cell0,cellf)
%INITIALIZE SIMULATION PARAMETER
nx = data.nx;
xpos_node = data.xpos_node;
N0 = length(cell0);
Nf = length(cellf);

cell0 = sort(cell0);
cellf = sort(cellf);

cell_vec0 = zeros(nx,1);
cell_vecf = zeros(nx,1);

%PAUL these could perhaps be kernels, but there are data dependencies, so atomics would likely be necessary unless we can break them out
for i = 1:N0
    cell_vec0(cell0(i)) = cell_vec0(cell0(i)) + 1;
end

for i = 1:Nf
    cell_vecf(cellf(i)) = cell_vecf(cellf(i)) + 1;
end

figure(1001)
plot(xpos_node,cell_vec0,'b');xlabel('x');ylabel('# of particles');title('# of Particles in cell');
hold on;
plot(xpos_node,cell_vecf,'g');xlabel('x');ylabel('# of particles');title('# of Particles in cell');
hold off;
