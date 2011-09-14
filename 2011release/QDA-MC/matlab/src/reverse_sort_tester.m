function reverse_sort_tester(N)
x0(1:N,1) = 0;
for i = 1:N
	x0(i) = round(100*rand);
end
[x,IX] = sort(x0);
xtemp = zeros(N,1);
for i = 1:N
    xtemp(IX(i)) = x((i));
end

[x0,x,xtemp]