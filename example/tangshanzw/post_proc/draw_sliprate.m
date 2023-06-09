% close all
clc;
clear;

set_mpath

% -------------------------- parameters input -------------------------- %

parfile    = '../run/params.json'
output_dir = '../run/output'

par = loadjson(parfile);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);

savegif = 0;
fnm_gif = 'slip_rate.gif';

%it_seq = 10 : 10 : 2000;
it_seq = 1999;

%---------------------------------------------------------------
%- read and plot
%---------------------------------------------------------------

nt = fault_num_time(output_dir);

[x,y,z] = gather_fault_coord(output_dir,nproj,nprok);

x = x * 1e-3;
y = y * 1e-3;
z = z * 1e-3;

x1 = x(j1:j2, k1:k2);
y1 = y(j1:j2, k1:k2);
z1 = z(j1:j2, k1:k2);

figure;

num_it_seq = length(it_seq);

for n = 1 : num_it_seq

  nlayer = it_seq(n)

% disp(it);
[Vs1,t] = gather_fault_1tlay(output_dir,nlayer,'Vs1',nproj,nprok);
[Vs2,t] = gather_fault_1tlay(output_dir,nlayer,'Vs2',nproj,nprok);

% V = Vs1(k1:k2, j1:j2);
% V = Vs2(k1:k2, j1:j2);
V = sqrt(Vs1(j1:j2,k1:k2).^2+Vs2(j1:j2,k1:k2).^2);
% i = ceil((nlayer-998) / 2);
% slip(i) = V(300,500);
% Vs = sqrt(Vs1.^2+Vs2.^2);
% surf(x1, y1, z1, V);
surf(x1, y1, z1, V);

axis equal; 
shading flat;
view([60, 30]);    
axis equal;
colormap( 'jet' );
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
xlabel('Hanging-wall Distance (km)','rotation',-48,'fontsize',9.4,'position',[45 -25]);
ylabel('Strike Distance (km)','rotation',15,'fontsize',9.4,'position',[40,20]);
zlabel('Depth (km)','fontsize',9.4);

% c.Position(2)=0.3;
set(gcf,'color','w');
title(['Total slip rate on fault',newline,'t=' num2str(t),'s'],'FontSize',12);

drawnow;
pause(0.5);
if savegif
   im=frame2im(getframe(gcf));
   [imind,map]=rgb2ind(im,256);
   if i==1
       imwrite(imind,map,fnm_gif,'gif','LoopCount',Inf,'DelayTime',0.5);
   else
       imwrite(imind,map,fnm_gif,'gif','WriteMode','append','DelayTime',0.5);
   end
end

end
