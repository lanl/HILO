function [phi,J,E,phi_S2] = node_moment_calc(x0,mu0,wx,xf,cell0,cellf,phi,J,E,phi_S2,data)
%INITIALIZATION OF SIMULATION PARAMETERS%%%%%%%%%%%%%%
dx = data.dx;
xpos_face = data.xpos_face;
lx = data.lx;
lenx = length(xf);
eps = data.eps;
epsd2 = eps/2;
%CALCULATION OF THE MOMENT QUANTITIES%%%%%%%%%%%%%%%%%
for i = 1:lenx %loop through all particles
    dcell = cellf(i) - cell0(i);
    if dcell == 0 %i.e. the particle didn't escape the cell
        if x0(i) ~= 0 && x0(i) ~= lx && xf(i) ~= 0 && xf(i) ~= lx
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/abs(mu0(i)))^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*abs(mu0(i));            
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/epsd2)^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*epsd2;                            
            end
            J(cell0(i)) = J(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*mu0(i)/abs(mu0(i));            
        elseif x0(i) == 0%entering from left boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/abs(mu0(i)))^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/epsd2)^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*epsd2;                
            end
            J(cell0(i)) = J(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*mu0(i)/abs(mu0(i));
        elseif x0(i) == lx%entering from right boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/abs(mu0(i)))^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/epsd2)^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*epsd2;                
            end
            J(cell0(i)) = J(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*mu0(i)/abs(mu0(i));
        elseif xf(i) == 0 %escaped from left boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/abs(mu0(i)))^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/epsd2)^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*epsd2;                
            end
            J(cell0(i)) = J(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*mu0(i)/abs(mu0(i));
        elseif xf(i) == lx %escaped from right boundary
            if abs(mu0(i)) >= eps
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/abs(mu0(i));
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/abs(mu0(i)))^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*abs(mu0(i));
            else
                phi(cell0(i)) = phi(cell0(i)) + wx(i)*abs(xf(i) - x0(i))/epsd2;
                phi_S2(cell0(i)) = phi_S2(cell0(i)) + (wx(i)*abs(xf(i) - x0(i))/epsd2)^2;
                E(cell0(i)) = E(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*epsd2;                
            end
            J(cell0(i)) = J(cell0(i)) + wx(i)*abs(xf(i) - x0(i))*mu0(i)/abs(mu0(i));
        end
    else %if the particle actually escaped the cell
        if dcell < 0%i.e. backward flight direction
            for j = cell0(i):-1:cellf(i)
                if j == cell0(i)
                    if abs(mu0(i)) >= eps
                        phi(j) = phi(j) + wx(i)*abs(x0(i)-xpos_face(j))/abs(mu0(i));
                        phi_S2(j) = phi_S2(j) + (wx(i)*abs(x0(i)-xpos_face(j))/abs(mu0(i)))^2;
                        E(j) = E(j) +  wx(i)*abs(x0(i)-xpos_face(j))*abs(mu0(i));
                    else
                        phi(j) = phi(j) + wx(i)*abs(x0(i)-xpos_face(j))/epsd2;
                        phi_S2(j) = phi_S2(j) + (wx(i)*abs(x0(i)-xpos_face(j))/epsd2)^2;
                        E(j) = E(j) +  wx(i)*abs(x0(i)-xpos_face(j))*epsd2;
                    end
                    J(j) = J(j) +  wx(i)*abs(x0(i)-xpos_face(j))*mu0(i)/abs(mu0(i));
                elseif j == cellf(i)
                    if abs(mu0(i)) >= eps
                        phi(j) = phi(j) +  wx(i)*abs(xf(i)-xpos_face(j+1))/abs(mu0(i));                        
                        phi_S2(j) = phi_S2(j) +  (wx(i)*abs(xf(i)-xpos_face(j+1))/abs(mu0(i)))^2;                        
                        E(j) = E(j) + wx(i)*abs(xf(i)-xpos_face(j+1))*abs(mu0(i));                        
                    else
                        phi(j) = phi(j) +  wx(i)*abs(xf(i)-xpos_face(j+1))/epsd2;                        
                        phi_S2(j) = phi_S2(j) +  (wx(i)*abs(xf(i)-xpos_face(j+1))/epsd2)^2;                        
                        E(j) = E(j) + wx(i)*abs(xf(i)-xpos_face(j+1))*epsd2;                                                
                    end
                    J(j) = J(j) + wx(i)*abs(xf(i)-xpos_face(j+1))*mu0(i)/abs(mu0(i));                        
                else
                    if abs(mu0(i)) >= eps
                        phi(j) = phi(j) + wx(i)*dx/abs(mu0(i));
                        phi_S2(j) = phi_S2(j) + (wx(i)*dx/abs(mu0(i)))^2;
                        E(j) = E(j) + wx(i)*dx*abs(mu0(i));
                    else
                        phi(j) = phi(j) + wx(i)*dx/epsd2;
                        phi_S2(j) = phi_S2(j) + (wx(i)*dx/epsd2)^2;
                        E(j) = E(j) + wx(i)*dx*epsd2;                        
                    end
                    J(j) = J(j) + wx(i)*dx*mu0(i)/abs(mu0(i));
                end
            end                
        else%i.e. forward flight direction
            if x0(i) == 0%if particle originated from left bound
                for j = cell0(i):cellf(i)
                    if j == cell0(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j+1)-x0(i))/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j+1)-x0(i))/abs(mu0(i)))^2;
                            E(j) = E(j) + wx(i)*abs(xpos_face(j+1)-x0(i))*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j+1)-x0(i))/epsd2;
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j+1)-x0(i))/epsd2)^2;
                            E(j) = E(j) + wx(i)*abs(xpos_face(j+1)-x0(i))*epsd2;                            
                        end
                        J(j) = J(j) + wx(i)*(xpos_face(j+1)-x0(i));
                    elseif j == cellf(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j)-xf(i))/abs(mu0(i));                        
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j)-xf(i))/abs(mu0(i)))^2;                        
                            E(j) = E(j) + wx(i)*abs(xpos_face(j)-xf(i))*abs(mu0(i));                        
                        else
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j)-xf(i))/epsd2;                        
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j)-xf(i))/epsd2)^2;                        
                            E(j) = E(j) + wx(i)*abs(xpos_face(j)-xf(i))*epsd2;                                                    
                        end
                        J(j) = J(j) + wx(i)*abs(xpos_face(j)-xf(i))*mu0(i)/abs(mu0(i));
                    else
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*dx/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + (wx(i)*dx/abs(mu0(i)))^2;
                            E(j) = E(j) + wx(i)*dx*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)*dx/epsd2;
                            phi_S2(j) = phi_S2(j) + (wx(i)*dx/epsd2)^2;
                            E(j) = E(j) + wx(i)*dx*epsd2;                            
                        end
                        J(j) = J(j) + wx(i)*dx*mu0(i)/abs(mu0(i));
                    end
                end
            else
                for j = cell0(i):cellf(i)
                    if j == cell0(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j+1)-x0(i))/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j+1)-x0(i))/abs(mu0(i)))^2;
                            E(j) = E(j) + wx(i)*abs(xpos_face(j+1)-x0(i))*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j+1)-x0(i))/epsd2;
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j+1)-x0(i))/epsd2)^2;
                            E(j) = E(j) + wx(i)*abs(xpos_face(j+1)-x0(i))*epsd2;                            
                        end
                        J(j) = J(j) + wx(i)*abs(xpos_face(j+1)-x0(i))*mu0(i)/abs(mu0(i));
                    elseif j == cellf(i)
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j)-xf(i))/abs(mu0(i));                        
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j)-xf(i))/abs(mu0(i)))^2;                        
                            E(j) = E(j) + wx(i)*abs(xpos_face(j)-xf(i))*abs(mu0(i));                        
                        else
                            phi(j) = phi(j) + wx(i)*abs(xpos_face(j)-xf(i))/epsd2;                        
                            phi_S2(j) = phi_S2(j) + (wx(i)*abs(xpos_face(j)-xf(i))/epsd2)^2;                        
                            E(j) = E(j) + wx(i)*abs(xpos_face(j)-xf(i))*epsd2;                                                    
                        end
                        J(j) = J(j) + wx(i)*abs(xpos_face(j)-xf(i))*mu0(i)/abs(mu0(i));
                    else
                        if abs(mu0(i)) >= eps
                            phi(j) = phi(j) + wx(i)*dx/abs(mu0(i));
                            phi_S2(j) = phi_S2(j) + (wx(i)*dx/abs(mu0(i)))^2;
                            E(j) = E(j) + wx(i)*dx*abs(mu0(i));
                        else
                            phi(j) = phi(j) + wx(i)*dx/epsd2;
                            phi_S2(j) = phi_S2(j) + (wx(i)*dx/epsd2)^2;
                            E(j) = E(j) + wx(i)*dx*epsd2;                            
                        end
                        J(j) = J(j) + wx(i)*dx*mu0(i)/abs(mu0(i));
                    end
                end                
            end
        end
    end
end