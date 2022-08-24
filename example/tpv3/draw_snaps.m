close all
clc;
clear;
addmypath;
parfile = 'params.json';
par = loadjson( parfile);
DT = par.DT;
dh = par.DH;
rho = par.rho1;
vs= par.vs1;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;
Faultgrid = par.Fault_grid;
j1 = Faultgrid(1);
j2 = Faultgrid(2);
k1 = Faultgrid(3);
k2 = Faultgrid(4);
NT = floor(TMAX/(DT*TSKP));
%calculate Mw
faultfile   = 'fault_coord.nc';
jac  = ncread(faultfile, 'jac');
D = gather_snap(parfile,'Us0',NT);
miu = rho*vs^2;
A = jac*dh*dh;
M0 = miu*sum(sum(D.*A));
Mw = roundn(2*log10(M0)/3.0-6.06,-2);

savegif = 1;
filename1 = ['Vs-ts-l-7.44.gif'];
its = 50:25:NT;
its = floor(its);
nt = length(its);

[x,y,z] = gather_coord(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;

x1 = x(j1:j2,k1:k2);
y1 = y(j1:j2,k1:k2);
z1 = z(j1:j2,k1:k2);

figure(1);
for i = 1  : nt
it = its(i);
disp(it);
Vs1 = gather_snap(parfile,'Vs1',it);
Vs2 = gather_snap(parfile,'Vs2',it);

% V = Vs2(j1:j2,k1:k2);
V = sqrt(Vs1(j1:j2,k1:k2).^2+Vs2(j1:j2,k1:k2).^2);

surf(x1, y1, z1, V );
axis equal; 
shading interp;
view([60, 30]);    
axis image;
axis equal;
colormap( 'jet' );
c = colorbar;
c = colorbar;
ax =gca;
ax.Position(1) = 0.15
ax.Position(3) = 0.72
axpos = ax.Position;
axpos=ax.Position 
c.Position(1)=0.90;
c.Position(2)=0.12;
c.Position(3) = 0.9*c.Position(3);
c.Position(4) = 0.74; 
xlabel('Prep-to-strike (km)','rotation',-48,'fontsize',9.4,'position',[45 -25]);
ylabel('Strike(N40°E) Distance (km)','rotation',15,'fontsize',9.4,'position',[40,20]);
zlabel('Depth (km)','fontsize',9.4);

% c.Position(2)=0.3;
set(gcf,'color','w');
title(['tangshan fault ',newline,'Mw = ',num2str(Mw),newline,'Vs t=' num2str(it*DT*TSKP),'s'],'FontSize',12)
% set(gca,'FontSize',12);
% set(gca,'Clim',[0,8]);

drawnow;
pause(0.5);
if savegif
   im=frame2im(getframe(gcf));
   [imind,map]=rgb2ind(im,256);
   if i==1
       imwrite(imind,map,filename1,'gif','LoopCount',Inf,'DelayTime',0.5);
   else
       imwrite(imind,map,filename1,'gif','WriteMode','append','DelayTime',0.5);
   end
end
end
