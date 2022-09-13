% close all
clc;
clear;
addmypath;

% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project2/params.json'
output_dir='../../project2/output'

par = loadjson(parfnm);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);

savegif = 0;
filename = ['Vs-ts-l-7.44.gif'];
nt = fault_num_time(output_dir);
[x,y,z] = gather_fault_coord(output_dir,nproj,nprok);

x = x * 1e-3;
y = y * 1e-3;
z = z * 1e-3;

x1 = x(k1:k2, j1:j2);
y1 = y(k1:k2, j1:j2);
z1 = z(k1:k2, j1:j2);

figure;
for nlayer = 10 : 10 : 1000
% disp(it);
[Vs1,t] = gather_fault(output_dir,nlayer,'Ts1',nproj,nprok);
% [Vs1,t] = gather_fault(output_dir,nlayer,'Vs1',nproj,nprok);
% [Vs2,t] = gather_fault(output_dir,nlayer,'Vs2',nproj,nprok);

V = Vs1(k1:k2, j1:j2);
% V = Vs2(k1:k2, j1:j2);
% V = sqrt(Vs1(k1:k2, j1:j2).^2+Vs2(k1:k2, j1:j2).^2);
i = ceil(nlayer / 10);
slip(i) = V(100,200);
% Vs = sqrt(Vs1.^2+Vs2.^2);
% surf(x1, y1, z1, V);
surf(x1, y1, z1, V);

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
%title(['tangshan fault ',newline,'Mw = ',num2str(Mw),newline,'Vs t=' num2str(t),'s'],'FontSize',12);
title(['tangshan fault ',newline,'Vs t=' num2str(t),'s'],'FontSize',12);
% set(gca,'FontSize',12);
% set(gca,'Clim',[0,12]);

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
