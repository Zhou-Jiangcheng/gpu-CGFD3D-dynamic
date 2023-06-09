 close all
clc;
clear;

set_mpath

% -------------------------- parameters input -------------------------- %

parfile    = '../run2/params.json'
output_dir = '../run2/output'

par = loadjson(parfile);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);

savegif = 0;
fnm_gif = 'slip.gif';

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
[slip1_all,t] = gather_fault_1tlay(output_dir,nlayer,'slip1',nproj,nprok);
[slip2_all,t] = gather_fault_1tlay(output_dir,nlayer,'slip2',nproj,nprok);
[slip_all ,t] = gather_fault_1tlay(output_dir,nlayer,'slip',nproj,nprok);

slip1 = slip1_all(j1:j2,k1:k2);
slip2 = slip2_all(j1:j2,k1:k2);
slip  = slip_all(j1:j2,k1:k2);
%V = sqrt(Vs1(j1:j2,k1:k2).^2+Vs2(j1:j2,k1:k2).^2);

slip3 = zeros(size(slip1));

hold off
surf(x1, y1, z1, slip);

%hold on
%skip_j = 20;
%skip_k = 20;
%slip_scal = 100;
%quiver3(   x1(1:skip_j:end,1:skip_k:end)+5, ...
%           y1(1:skip_j:end,1:skip_k:end), ...
%           z1(1:skip_j:end,1:skip_k:end), ...
%        slip1(1:skip_j:end,1:skip_k:end)*slip_scal, ...
%        slip2(1:skip_j:end,1:skip_k:end)*slip_scal, ...
%        slip3(1:skip_j:end,1:skip_k:end)*slip_scal);

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
%xlabel('Hanging-wall Distance (km)','rotation',-48,'fontsize',9.4,'position',[45 -25]);
ylabel('Strike Distance (km)','rotation',15,'fontsize',9.4,'position',[40,20]);
zlabel('z-axis (km)','fontsize',9.4);

% c.Position(2)=0.3;
set(gcf,'color','w');
title(['Total slip on fault',newline,'t=' num2str(t),'s'],'FontSize',12);

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
