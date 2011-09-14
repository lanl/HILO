function sim_fig_plot(data,phi,J,E)

%INITIALIZATION OF SIMULATION PARAMETER
xpos_node = data.xpos_node;

%PLOT THE MOMENTS
figure(1);
subplot(3,1,1),plot(xpos_node,phi,'r');xlabel('x');ylabel('\phi');title('scalar flux');
subplot(3,1,2),plot(xpos_node,J,'b');xlabel('x');ylabel('J');title('current');
subplot(3,1,3),plot(xpos_node,E,'g');xlabel('x');ylabel('E');title('Eddington Tensor');