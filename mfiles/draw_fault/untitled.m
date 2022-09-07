%plot slip and init_t0
clear all;
close all;
clc;
addmypath;

% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project1/params.json'
output_dir='../../project1/output'

par = loadjson(parfnm);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);
nt = fault_num_time(output_dir);

[x,y,z] = gather_fault_coord(output_dir,nproj,nprok);

x = x * 1e-3;
y = y * 1e-3;
z = z * 1e-3;

x1 = x(k1:k2, j1:j2);
y1 = y(k1:k2, j1:j2);
z1 = z(k1:k2, j1:j2);
varnm1 = 'slip1';
varnm2 = 'slip2';
varnm3 = 'init_t0';
[Us1] = gather_fault(output_dir,nt,varnm1,nproj,nprok);
[Us2] = gather_fault(output_dir,nt,varnm2,nproj,nprok);
[t0] = gather_fault_final(output_dir,varnm3,nproj,nprok);

x1 = x(k1:k2, j1:j2);
y1 = y(k1:k2, j1:j2);
z1 = z(k1:k2, j1:j2);
t1 = t0(k1:k2, j1:j2);
% figure control parameters
flag_print = 0;
% get time contour
vec_t = 1:1:12;

%%
close all
% draw the rupture time
figure(1)
surf(x1, y1, z1, Us1(k1:k2,j1:j2) ); 

axis equal; 
shading interp;
view([60, 30]);
colormap( 'jet' );
colorbar
c = colorbar;
ax =gca;
ax.Position(1) = 0.15
ax.Position(3) = 0.72
axpos = ax.Position;
axpos=ax.Position 
c.Position(1)=0.905;
c.Position(2)=0.12;
c.Position(3) = 0.9*c.Position(3);
c.Position(4) = 0.74; 
xlabel('Prep-to-strike (km)','rotation',-48,'fontsize',9.4,'position',[45 -25]);
ylabel('Strike(N40°E) Distance (km)','rotation',15,'fontsize',9.4,'position',[40,20]);
zlabel('Depth (km)','fontsize',9.4);
title(['tangshan fault',newline,'slip along Strike'],'FontSize',12);
% set(gca,'FontSize',12);
set(gcf,'color','w');
% save and print figure

if flag_print==1
    width= 500;
    height=500;
    set(gcf,'paperpositionmode','manual');
    set(gcf,'paperunits','points');
    set(gcf,'papersize',[width,height]);
    set(gcf,'paperposition',[0,0,width,height]);
    fnm_out=['ts-l-strike'];
    print(gcf,[fnm_out '.png'],'-dpng');
end
%%
figure(2)
surf(x1, y1, z1, Us2(k1:k2,j1:j2) ); 
axis equal; 
shading interp;
view([60, 30]);
colormap( 'jet' );
colorbar; 
colorbar
c = colorbar;
ax =gca;
ax.Position(1) = 0.15
ax.Position(3) = 0.72
axpos = ax.Position;
axpos=ax.Position 
c.Position(1)=0.905;
c.Position(2)=0.12;
c.Position(3) = 0.9*c.Position(3);
c.Position(4) = 0.74; 
xlabel('Prep-to-strike (km)','rotation',-45,'fontsize',9.4,'position',[45 -25]);
ylabel('Strike(N40°E) Distance (km)','rotation',15,'fontsize',9.4,'position',[40,20]);
zlabel('Depth (km)','fontsize',9.4);
title(['tangshan fault',newline,'slip along Dip'],'FontSize',12);
% set(gca,'FontSize',12);
set(gcf,'color','w');
% save and print figure
if flag_print==1
    width= 500;
    height=500;
    set(gcf,'paperpositionmode','manual');
    set(gcf,'paperunits','points');
    set(gcf,'papersize',[width,height]);
    set(gcf,'paperposition',[0,0,width,height]);
    fnm_out=['ts-l-dip'];
    print(gcf,[fnm_out '.png'],'-dpng');
end
