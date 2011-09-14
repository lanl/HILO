function [x,mu] = xmu_sort(x,mu)

[x,IX] = sort(x);
mutemp = zeros(length(x),1);
for i = 1:length(x)
    mutemp(i) = mu(IX(i));
end
mu = mutemp;