function [f_avg,f_filter_flag] = f_filter(f,data,face_flag,E_flag,f_filter_flag)
%face_flag = flag that checks if the quantity is defined at face or cell
%E_flag = flag that hecks if the quantity being filtered is Eddington
%Tensor or not.
%center
%INITIALIZE SIMULATION PARAMETERS
nx = data.nx;
if E_flag == 1%Eddington tensor is being filtered
    if face_flag == 1%quantity is defined as face value
        f_avg = zeros(nx+1,1);
        for i = 1:nx+1
            if f(i) >= 0.33
                f_avg(i) = f(i);
            elseif i == 1
%                 f_avg(i) = f(i);
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i+1);
            elseif i == nx + 1
%                 f_avg(i) = f(i);
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i-1);
            else
                f_avg(i) = (1/4)*f(i-1) + (1/2)*f(i) + (1/4)*f(i+1);
            end
        end
        f_filter_flag = 0;%dummy variable
    elseif face_flag == 0%quantity is defined at node value
        f_filter_flag = zeros(nx,1);
        f_avg = zeros(nx,1);
        for i = 1:nx
            if f(i) >= 0.33
                f_avg(i) = f(i);
            elseif i == 1
%                 f_avg(i) = f(i);
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i+1);
                f_filter_flag(i) = 1;
            elseif i == nx
%                 f_avg(i) = f(i);
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i-1);
                f_filter_flag(i) = 1;
            else
                f_avg(i) = (1/4)*f(i-1) + (1/2)*f(i) + (1/4)*f(i+1);
                f_filter_flag(i) = 1;
            end
        end
    end
else %phi LO is getting filtered
    f_avg = zeros(nx,1);
    for i = 1:nx
        if f_filter_flag(i) == 0
            f_avg(i) = f(i);
            %DO NOT FILTER IF  FILTER FLAG IS OFF FOR THE NODE
        else            
            if i == 1
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i+1);
            elseif i == nx
                f_avg(i) = (3/4)*f(i) + (1/4)*f(i-1);
            else
                f_avg(i) = (1/4)*f(i-1) + (1/2)*f(i) + (1/4)*f(i+1);
            end
        end
    end   
    f_filter_flag = 0;%dummy variable
end