function [phi,J,E,phi_S2] = face_moment_calc(x0,mu0,wx,xf,cell0,cellf,phi,J,E,phi_S2,data)
%INITIALIZATION OF SIMULATION PARAMETERS%%%%%%%%%%%%%%
nx = data.nx;
eps = data.eps;%This is the angular limit
epsd2 = eps/2;
lx = data.lx;
lenx = length(xf);
%CALCULATION OF THE MOMENT QUANTITIES%%%%%%%%%%%%%%%%%
for i = 1:lenx %loop through all particles
    dcell = cellf(i) - cell0(i);
    if dcell == 0 %i.e. the particle didn't escape the cell
        if x0(i) ~= 0 && x0(i) ~= lx && xf(i) ~= 0 && xf(i) ~= lx
        elseif x0(i) == 0%entering from left boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + wx(i)*wx(i)/mu0(i)/mu0(i);
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + wx(i)*wx(i)/epsd2/epsd2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*epsd2;
            end
            J(cell0(i)) = J(cell0(i)) + wx(i);
        elseif x0(i) == lx%entering from right boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)+1) = phi(cell0(i)+1) + wx(i)/abs(mu0(i));
                phi_S2(cell0(i)+1) = phi_S2(cell0(i)+   1) + wx(i)*wx(i)/mu0(i)/mu0(i);
                E(cell0(i)+1) = E(cell0(i)+1) + wx(i)*abs(mu0(i));
%RESUME2
            else
                phi(cell0(i)+1) = phi(cell0(i)+1) + wx(i)/(epsd2);
                phi_S2(cell0(i)+1) = phi_S2(cell0(i)+1) + wx(i)*wx(i)/epsd2/epsd2;
                E(cell0(i)+1) = E(cell0(i)+1) + wx(i)*epsd2;
            end
            J(cell0(i)+1) = J(cell0(i)+1) - wx(i);            
        elseif xf(i) == 0 %escaped from left boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + wx(i)*wx(i)/mu0(i)/mu0(i);
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)/(epsd2);
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + wx(i)*wx(i)/epsd2/epsd2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*epsd2;
            end
            J(cell0(i)) = J(cell0(i)) - wx(i);            
        elseif xf(i) == lx %escaped from right boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)+1) = phi(cell0(i)+1) + wx(i)/abs(mu0(i));
                phi_S2(cell0(i)+1) = phi_S2(cell0(i)+1) + wx(i)*wx(i)/mu0(i)/mu0(i);
                E(cell0(i)+1) = E(cell0(i)+1) + wx(i)*abs(mu0(i));
            else
                phi(cell0(i)+1) = phi(cell0(i)+1) + wx(i)/(epsd2);
                phi_S2(cell0(i)+1) = phi_S2(cell0(i)+1) + wx(i)*wx(i)/epsd2/epsd2;
                E(cell0(i)+1) = E(cell0(i)+1) + wx(i)*epsd2;
            end
            J(cell0(i)+1) = J(cell0(i)+1) + wx(i);
        end
    else %if the particle actually escaped the cell
        if dcell < 0%i.e. backward flight direction
            for j = cell0(i):-1:cellf(i)
                if j == cell0(i)
                    if abs(mu0(i)) >= eps
                        phi(j) = phi(j) + wx(i)/abs(mu0(i));
                        phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                        E(j) = E(j) + wx(i)*abs(mu0(i));
                    else
                        phi(j) = phi(j) + wx(i)/(epsd2);
                        phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                        E(j) = E(j) + wx(i)*epsd2;
                    end
                    J(j) = J(j) - wx(i);
                elseif j == cellf(i)
                else
                    if abs(mu0(i)) >= eps
                        phi(j) = phi(j) + wx(i)/abs(mu0(i));
                        phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                        E(j) = E(j) + wx(i)*abs(mu0(i));
                    else
                        phi(j) = phi(j) + wx(i)/(epsd2);
                        phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                        E(j) = E(j) + wx(i)*epsd2;
                    end
                    J(j) = J(j) - wx(i);
                end
            end                
            if xf(i) == 0
                if abs(mu0(i)) >= eps
                    phi(1) = phi(1) + wx(i)/abs(mu0(i));
                    phi_S2(1) = phi_S2(1) + wx(i)*wx(i)/mu0(i)/mu0(i);
                    E(1) = E(1) + wx(i)*abs(mu0(i));
                else
                    phi(1) = phi(1) + wx(i)/(epsd2);
                    phi_S2(1) = phi_S2(1) + wx(i)*wx(i)/epsd2/epsd2;
                    E(1) = E(1) + wx(i)*epsd2;
                end
                J(1) = J(1) - wx(i);
            end
        else%i.e. forward flight direction
            if x0(i) == 0%if particle originated from left bound
                for j = cell0(i):cellf(i)
                    if j == cell0(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                            E(j) = E(j) + wx(i)*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)/(epsd2);
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                            E(j) = E(j) + wx(i)*epsd2;
                        end
                        J(j) = J(j) + wx(i);                        
                    elseif j == cellf(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                            E(j) = E(j) + wx(i)*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)/(epsd2);
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                            E(j) = E(j) + wx(i)*epsd2;
                        end
                        J(j) = J(j) + wx(i);
                    else
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                            E(j) = E(j) + wx(i)*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)/(epsd2);
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                            E(j) = E(j) + wx(i)*epsd2;
                        end
                        J(j) = J(j) + wx(i);
                    end
                end
            else
                for j = cell0(i):cellf(i)
                    if j == cell0(i)
                    elseif j == cellf(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                            E(j) = E(j) + wx(i)*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)/(epsd2);
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                            E(j) = E(j) + wx(i)*epsd2;
                        end
                        J(j) = J(j) + wx(i);
                    else
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/mu0(i)/mu0(i);
                            E(j) = E(j) + wx(i)*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)/(epsd2);
                            phi_S2(j) = phi_S2(j) + wx(i)*wx(i)/epsd2/epsd2;
                            E(j) = E(j) + wx(i)*epsd2;
                        end
                        J(j) = J(j) + wx(i);
                    end
                end                
            end
            if xf(i) == lx
                if abs(mu0(i)) >= eps
                    phi(nx+1) = phi(nx+1) + wx(i)/abs(mu0(i));
                    phi_S2(nx+1) = phi_S2(nx+1) + wx(i)*wx(i)/mu0(i)/mu0(i);
                    E(nx+1) = E(nx+1) + wx(i)*abs(mu0(i));
                else
                    phi(nx+1) = phi(nx+1) + wx(i)/(epsd2);
                    phi_S2(nx+1) = phi_S2(nx+1) + wx(i)*wx(i)/epsd2/epsd2;
                    E(nx+1) = E(nx+1) + wx(i)*epsd2;
                end
                J(nx+1) = J(nx+1) + wx(i);
            end
        end
    end
end
