function [xtemp] = reverse_sort(x,IX)
N = length(x);
xtemp = zeros(N,1);
for i = 1:N
    xtemp(IX(i)) = x((i));
end