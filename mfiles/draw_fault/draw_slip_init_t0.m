%plot slip and init_t0
clc;
clear all;
close all;
addmypath;
parfile = 'params.json';
par = loadjson(parfile);
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;
Fault_grid = par.Fault_grid; 
DT = par.DT;
dh = par.DH;
rho = par.rho1;
vs= par.vs1;
NT = floor(TMAX/(DT*TSKP));
j1 = Fault_grid(1);
j2 = Fault_grid(2);
k1 = Fault_grid(3);
k2 = Fault_grid(4);
NT = floor(TMAX/(DT*TSKP));
%calculate Mw
faultfile   = 'fault_coord.nc';
jac  = ncread(faultfile, 'jac');
D = gather_snap(parfile,'Us0',NT);
miu = rho*vs^2;
A = jac*dh*dh;
M0 = miu*sum(sum(D.*A));
Mw = roundn(2*log10(M0)/3.0-6.06,-2);

[x,y,z] = gather_coord(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;

x1 = x(j1:j2,k1:k2);
y1 = y(j1:j2,k1:k2);
z1 = z(j1:j2,k1:k2);

Us1 = gather_snap(parfile,'Us1',NT);
Us2 = gather_snap(parfile,'Us2',NT);

t = gather_snap(parfile,'init_t0');
t1 = t(j1:j2 , k1:k2);
% figure control parameters
flag_print = 0;
% get time contour
vec_t = 1:1:25;

h0 = figure;
[C1,h] = contour(x1,z1,t1,vec_t,'k','ShowText','on');
clabel(C1,h,'labelSpacing',10000);
clabel(C1,h,'fontsize',14);
set(h0,'visible','off');

h0 = figure;
[C2,h] = contour(y1,z1,t1,vec_t,'k','ShowText','on');
clabel(C2,h,'labelSpacing',10000); 
clabel(C2,h,'fontsize',14);
set(h0, 'visible', 'off');

% find of t = 2*n, n = 1,2,3... 

[~,c] = find(rem(C1(1,:),1)==0);
count = size(c,2);

n = C1(2,c); 

C1(:,c)=[]; 
C2(:,c)=[]; 

t_x = C1(1,:); 
t_y = C2(1,:);
t_z = C1(2,:);
%%
close all
% draw the rupture time
figure(1)
surf(x1, y1, z1, Us1(j1:j2,k1:k2) ); 
hold on;
temp = 0;
for ii = 1:2
    plot3(t_x(temp+1:temp+n(ii)),t_y(temp+1:temp+n(ii)),t_z(temp+1:temp+n(ii)),'r','linewidth',1.0);
    temp = temp + n(ii);
    hold on;
end
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
surf(x1, y1, z1, Us2(j1:j2,k1:k2) ); 
hold on;
temp = 0;
for ii = 1:count
    plot3(t_x(temp+1:temp+n(ii)),t_y(temp+1:temp+n(ii)),t_z(temp+1:temp+n(ii)),'r','linewidth',1.0);
    temp = temp + n(ii);
    hold on;
end
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
