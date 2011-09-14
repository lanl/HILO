function monte_carlo(NP,lx,nx,tol,tol_std,fig_upd,witer)
%NP = reference particle number
%lx = length of system
%nx = number of cells in the system
%tol = relative difference tolerance for convergence
%witer = # of iteration to wait until actual averaging starts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INITIALIZATION OF VITAL SIMULATION PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n0 = 0.01;%base density parameter
dx = lx/nx;%cell width
eps = 0.1;%the threshold angle for tallying
sig_t = 1.;%total collision cross-section
sig_s = 0.99*sig_t;%scattering collision cross-section
D = sig_s/(3*sig_t*sig_t);%diffusion coefficient
prob_s = sig_s/sig_t;%probability of scattering
Jp_l = 1;%The partial current on the left face
Q0(1:nx,1) = 1.0;%The source term uniform for now
% Q0(1:nx+1,1) = 1.0;%The source term uniform for now
Q0_LO = 1.0;
NPc = round(NP/nx);%number of particles in cell 
NP_tot = NP;%total number of particles for the QDAMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FORM DATA STRUCTURE FOR SIMULATION PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = struct('NP',NP,'lx',lx,'nx',nx,'dx',dx,'n0',n0,'sig_t',sig_t,...
              'sig_s',sig_s,'prob_s',prob_s,'Q0',Q0,'Jp_l',Jp_l,'eps',eps,...
              'Q0_LO',Q0_LO,'D',D,'NP_tot',NP_tot,'NPc',NPc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATION OF INITIAL CONDITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data] = grid_position_calculation(data);%calculate the cell center and face position
xpos_node = data.xpos_node;xpos_face = data.xpos_face;
% [x,mu] = placement_func(data,n);
lbc_flag = input('What B.C. do you want on the left surface? 0 = isotropic source, 1 = beam source, 2 = vacuum \n');
data.lbc_flag = lbc_flag;
%%%%%%%%%%%%%%%%%%%%%%%%%%o%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PLOT AND PRINT INITIAL CONDITION AS WELL AS SIMULATION PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_param_print(data);
fprintf('n');fprintf('Please press any key to continue with simulation \n');
pause
%%%%%%%%%%%%%%%%%%%%%%%%%%o%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INITIALIZE THE AVERAGE VALUES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_cnt = 0;%initialize the figure update count
iter = 0;%tally for iteration for convergence
iter_avg = 0;%iteration for averaging
phi_tot = 0;%tally for average phi
phi_n_tot = 0;
phi_LO_tot = 0;%tally for average lower order phi
J_tot = 0;%tally for average J
J_n_tot = 0;
E_tot = 0;%tally for average E
E_n_tot = 0;
phi_avg = zeros(nx+1,1);
phi_avg_old = phi_avg;
E_n_avg = zeros(nx,1);
J_avg = zeros(nx+1,1);
E_avg = zeros(nx+1,1);
phi_LO_avg = zeros(nx,1);
phi_LO_avg_old = phi_LO_avg;
stdev_vec = zeros(1000,1);
% rel_diff_vec = 0;
%INITIALIZE THE LOWER ORDER SCALAR FLUX SOLUTION USING DIFFUSION APPROXIMATION
E_HO(1:nx+1,1) = 1/3;
E_HO_n(1:nx,1) = 1/3;
J_HO(1:nx+1,1) = 0;
phi_HO(1:nx+1,1) = 0;
phi_HO_S1 = zeros(nx+1,1);%mean tally for flux
phi_HO_S1_n = zeros(nx,1);%mean tally for flux at node
phi_HO_S2 = zeros(nx+1,1);%mean of square tally for flux
phi_HO_S2_n = zeros(nx,1);%mean of square tally for flux
[phi_LO] = LO_solver(data,phi_HO,J_HO,E_HO,E_HO_n,1);
flag_sim = 0;
avg_rel_diff = 0;
tal_rda = 0;
iter_vec = zeros(1000,1);
rel_diff_vec(1:1000,1) = 1.3;%dummy value
rel_diff_LO_vec(1:1000,1) = 1.3;%dummy value
samp_cnt = 0;%total history tally
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATION STARTS HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
while flag_sim == 0;
    iter = iter + 1;
    fprintf('This the %g',iter);fprintf('th iteration\n');
    fig_cnt = fig_cnt + 1;
    if iter == 1
        Qbar = Q0 + sig_s*phi_LO;
        [phi,J,E,phi_HO_S2,phi_n,J_n,E_n,phi_HO_S2_n] = collision_and_tally(data,Qbar,phi_HO_S2,phi_HO_S2_n);
        [E_n,f_filter_flag] = f_filter(E_n,data,0,1,0);
        E = f_filter(E,data,1,1,0);
        [phi_LO] = LO_solver(data,phi,J,E,E_n,0);
        phi_LO = f_filter(phi_LO,data,0,0,f_filter_flag);
    else
        Qbar = Q0 + sig_s*phi_LO_avg;
        [phi,J,E,phi_HO_S2,phi_n,J_n,E_n,phi_HO_S2_n] = collision_and_tally(data,Qbar,phi_HO_S2,phi_HO_S2_n);
        [E_n,f_filter_flag] = f_filter(E_n,data,0,1,0);
        E = f_filter(E,data,1,1,0);
        [phi_LO] = LO_solver(data,phi_avg,J_avg,E_avg,E_n_avg,0);
        phi_LO = f_filter(phi_LO,data,0,0,f_filter_flag);
    end
    %AVERAGE QUANTITIES
    if iter >= witer
        tal_rda = tal_rda + 1;
        iter_avg = iter_avg + 1;
        samp_cnt = samp_cnt + NP_tot;
        phi_HO_S1 = phi_HO_S1 + phi;
        phi_HO_S1_n = phi_HO_S1_n + phi_n;
        [phi_mean,phi_std] = mean_std_calc(phi_HO_S1,phi_HO_S2,samp_cnt,data);
        [phi_n_mean,phi_n_std] = mean_std_calc(phi_HO_S1_n*dx,phi_HO_S2_n,samp_cnt,data);
        fprintf('The maximum standard deviation of flux is, max(sig_phi) = %g \n',max(phi_std));
        fprintf('The maximum standard deviation of flux at node is, max(sig_phi) = %g \n',max(phi_n_std));
        phi_tot = phi_tot + phi;phi_avg = phi_tot/iter_avg;
        phi_n_tot = phi_n_tot + phi_n;phi_n_avg = phi_n_tot/iter_avg;
        phi_LO_tot = phi_LO_tot + phi_LO;phi_LO_avg = phi_LO_tot/iter_avg;
        J_tot = J_tot + J;J_avg = J_tot/iter_avg;
        J_n_tot = J_n_tot + J_n;J_n_avg = J_n_tot/iter_avg;
        E_tot = E_tot + E;E_avg = E_tot/iter_avg;
        E_n_tot = E_n_tot + E_n;E_n_avg = E_n_tot/iter_avg;
        rel_diff = rel_diff_calc(data,phi_avg,phi_avg_old);
        rel_diff_LO = rel_diff_calc(data,phi_LO_avg,phi_LO_avg_old);
        avg_rel_diff = avg_rel_diff + rel_diff;
        iter_vec(iter_avg,1) = iter_avg;
        stdev_vec(iter_avg) = max(phi_n_std);
        rel_diff_vec(iter_avg,1) = rel_diff;
        rel_diff_LO_vec(iter_avg,1) = rel_diff_LO;
        fprintf('The relative difference of scalar flux is, %g \n',rel_diff);
        phi_avg_old = phi_avg;
        phi_LO_avg_old = phi_LO_avg;
        if max(phi_n_std) <= tol_std
            flag_sim = 1;
        end
        if tal_rda == 10
            avg_rel_diff = avg_rel_diff/tal_rda;
            tal_rda = 0;
            if avg_rel_diff <= tol
%                 flag_sim = 1;
            end
            avg_rel_diff = 0;
        end
%         if rel_diff <= tol %|| rel_diff_LO <= tol
%             flag_sim = 1;
%         end        
    end
    if fig_cnt == fig_upd%FIGURE PLOT
        figure(2);
        subplot(2,2,1),plot(xpos_face,phi_avg,'b');xlabel('x');ylabel('\phi^{ HO}');title('Scalar Flux');hold on;
        subplot(2,2,1),plot(xpos_node,phi_n_avg,'r');hold off;
        subplot(2,2,2),plot(xpos_face,J_avg,'b');xlabel('x');ylabel('J^{ HO}');title('Current');hold on;
        subplot(2,2,2),plot(xpos_node,J_n_avg,'r');hold off;
        subplot(2,2,3),plot(xpos_face,E_avg,'b');xlabel('x');ylabel('E^{ HO}');title('Eddington Tensor');hold on;
        subplot(2,2,3),plot(xpos_node,E_n_avg,'r');hold off;
        subplot(2,2,4),plot(xpos_node,phi_LO_avg,'r');xlabel('x');ylabel('\phi^{ LO}');title('LO Scalar Flux');
        figure(3);
        semilogy(iter_vec,rel_diff_vec,'r');xlabel('cycle');ylabel('(\phi^{n} - \phi^{n-1})/\phi^{n}');hold on;
        semilogy(iter_vec,rel_diff_LO_vec,'g');xlabel('cycle');ylabel('(\phi^{n} - \phi^{n-1})/\phi^{n}'); hold off;
        figure(4);
        plot(iter_vec,stdev_vec,'b');xlabel('iteration');ylabel('\sigma_{\phi}');title('Maximum Standard Deviation of \phi vs. Iteration')
        fig_cnt = 0;
    end
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PRINT OUT AND POST PROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter_vec(iter_vec == 0) = [];
rel_diff_vec(rel_diff_vec == 1.3) = [];
rel_diff_LO_vec(rel_diff_LO_vec == 1.3) = [];
stdev_vec(stdev_vec == 0) = [];
figure(2);
subplot(2,2,1),plot(xpos_node,phi_n_avg,'b');xlabel('x');ylabel('\phi^{HO}');title('Scalar Flux');
subplot(2,2,2),plot(xpos_node,J_n_avg,'r');xlabel('x');ylabel('J^{HO}');title('Current');
subplot(2,2,3),plot(xpos_node,E_n_avg,'g');xlabel('x');ylabel('E^{HO}');title('Eddington Tensor');
subplot(2,2,4),plot(xpos_node,phi_LO_avg,'r');xlabel('x');ylabel('\phi^{LO}');title('LO Scalar Flux');
figure(3);
semilogy(iter_vec,rel_diff_vec,'r');xlabel('cycle');ylabel('(\phi^{n} - \phi^{n-1})/\phi^{n}');hold on;title('relative difference vs. cycle QDAMC');
semilogy(iter_vec,rel_diff_LO_vec,'g');xlabel('cycle');ylabel('(\phi^{n} - \phi^{n-1})/\phi^{n}');hold off;
[phi_HO_n] = nodal_value_calc(phi_avg,data);
rel_diff_HO_LO = abs((phi_n_avg - phi_LO_avg)./phi_n_avg);
figure(4);
plot(xpos_node,rel_diff_HO_LO,'b');xlabel('x');ylabel('(\phi_{QDAMC} - \phi_{LO})/\phi_{QDAMC}');title('relative-difference in \phi^{QDAMC}_{HO} and \phi^{QDAMC}_{LO}');
figure(5);
plot(iter_vec,stdev_vec,'b');xlabel('iteration');ylabel('\sigma_{\phi}');title('Maximum Standard Deviation of \phi');
fprintf('The total iteration required for convergence was, %g',iter);
fprintf('\n');
csvwrite('phi_avg.csv',phi_avg);
csvwrite('J_avg.csv',J_avg);
csvwrite('E_avg.csv',E_avg);
csvwrite('phiLO_avg.csv',phi_LO_avg);
csvwrite('rel_diff_HO_LO.csv',rel_diff_HO_LO);