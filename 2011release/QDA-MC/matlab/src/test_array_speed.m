function test_array_speed(N,nx)

x = zeros(N*nx,1);
tic
for j = 1:nx
    for i = 1:N
        x(i + N*(j-1)) = i;
    end
end
toc