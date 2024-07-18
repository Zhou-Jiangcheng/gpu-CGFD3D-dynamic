close all
clc;
clear;
addmypath;

% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/test.json'
output_dir='../../project/output'

id=1;

savegif = 0;
flag_km = 1;

par = loadjson(parfnm);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);
fault_index = par.fault_x_index;
nt = fault_num_time(output_dir);
[x,y,z] = gather_fault_coord(output_dir,fault_index(id),nproj,nprok);

if flag_km == 1
  x = x * 1e-3;
  y = y * 1e-3;
  z = z * 1e-3;
end

x1 = x(k1:k2, j1:j2);
y1 = y(k1:k2, j1:j2);
z1 = z(k1:k2, j1:j2);

% Vs, Vs1, Vs2, Tn, Ts1, Ts2
% Slip, Slip1, Slip2
% n->normal s1->strike s2->dip
var1 = "Vs1";
var2 = "Vs2";
figure(1);
for nlayer = 1001 : 100 : 1001
% disp(it);
% [Tn,t] = gather_fault(output_dir,fault_index(id),nlayer,'Tn',nproj,nprok);
[Vs1,t] = gather_fault(output_dir,fault_index(id),nlayer,var1,nproj,nprok);
[Vs2,t] = gather_fault(output_dir,fault_index(id),nlayer,var2,nproj,nprok);

% V = Tn(k1:k2, j1:j2);
V = Vs1(k1:k2, j1:j2);
% V = sqrt(Vs1(k1:k2, j1:j2).^2+Vs2(k1:k2, j1:j2).^2);
surf(x1, y1, z1, V);
set(gca,'layer','top');
set(gcf,'color','white','renderer','painters');
axis equal; 
shading interp;
view([60, 30]);    
xlabel('Prep-to-strike (km)','rotation',-48,'fontsize',9.4);
ylabel('Strike(N40°E) Distance (km)','rotation',15,'fontsize',9.4);
zlabel('Depth (km)','fontsize',9.4);

set(gcf,'color','w');
title(['tangshan fault ',newline,newline,'Vs t=' num2str(t),'s'],'FontSize',12);
% set(gca,'Clim',[0,12]);
colorbar;
colormap( 'jet' );

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
