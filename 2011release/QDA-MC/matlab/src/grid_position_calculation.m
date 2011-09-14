function [data] = grid_position_calculation(data)
%%%%%%%%%%%%Initialization of variables from data structure%%%%%%%%%%%%%%%%
dx = data.dx;
nx = data.nx;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Initialization of Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xpos_face(nx+1,1) = 0;
xpos_node(nx,1) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%Looping of Face and Node Position%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%Storing Position into Data Structure%%%%%%%%%%%%%%%%%%%%%%
data.xpos_face = xpos_face;
data.xpos_node = xpos_node;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%