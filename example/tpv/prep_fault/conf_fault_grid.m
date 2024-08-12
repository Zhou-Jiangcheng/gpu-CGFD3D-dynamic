% this script is generate fault grid, include strong boundary
% ny,nz is determined by script fault_index_length.m
clc;
clear;
close all;

addmypath;
nj =800; % without ghost points 
nk =600; 
dh = 100; %grid physics length
j1 = 50;
j2 = 750;
k1 = 50;
k2 = 550;

x = zeros(nj, nk);
y = zeros(nj, nk);
z = zeros(nj, nk);
id=1;
for k = 1:nk
  for j = 1:nj
        x(j,k) = 0;
        y(j,k) = (j-nj/2)*dh;
%         y(j,k) = (j-1)*dh;
        z(j,k) = (k-nk)*dh;
  end
end
if 0
% get random topography
N = [nj nk]; % size in pixels of image
F = 10;        % frequency-filter width
[Y,Z] = ndgrid(1:N(1),1:N(2));
j = min(Y-1,N(1)-Y+1);
k = min(Z-1,N(2)-Z+1);
H = exp(-.5*(j.^2+k.^2)/F^2);
X = real(ifft2(H.*fft2(randn(N))));
X = X .* 1e3;
x=x+X;
end
if 0
for j = 1:nj
    for k = 1:nk
        r1 = sqrt((y(j,k)-10e3).^2 + (z(j,k)+7.5e3).^2);
        r2 = sqrt((y(j,k)+10e3).^2 + (z(j,k)+7.5e3).^2);
        fxy = 0;
        if(r1 <3e3)
            fxy = 300 * (1+cos(pi*r1/3e3));
        end
        
        if(r2 <3e3)
            fxy = 300 * (1+cos(pi*r2/3e3));
        end
        
        x(j,k)=x(j,k)+fxy;
    end
end
end
figure
pcolor(y, z, x);
shading flat
colorbar
axis image


disp('calculating metric and base vectors...')
metric = cal_metric(x,y,z,dh);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);
jac = metric.jac;
disp('write output...')
fnm_out = sprintf("./fault_coord_%d.nc",id);
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'nj',nj);
dimid(2) = netcdf.defDim(ncid,'nk',nk);
dimid3(1) = netcdf.defDim(ncid, 'dim', 3);
dimid3(2) = dimid(1);
dimid3(3) = dimid(2);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
varid(4) = netcdf.defVar(ncid,'vec_n','NC_FLOAT',dimid3);
varid(5) = netcdf.defVar(ncid,'vec_m','NC_FLOAT',dimid3);
varid(6) = netcdf.defVar(ncid,'vec_l','NC_FLOAT',dimid3);
varid(7) = netcdf.defVar(ncid,'jac','NC_FLOAT',dimid);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.putVar(ncid,varid(7),jac);
netcdf.close(ncid);

if 1
figure;
surf(x/1e3, y/1e3, z/1e3); 
% surf(xf_inv, yf_inv, zf_inv);
shading interp;
title('Tangshan fault');
xlabel('x axis(km)');
ylabel('y axis(km)');
zlabel('depth(km)');
axis equal;
view([60, 30]);
set(gca,'FontName','Times New Roman','FontSize',15,'LineWidth',1.5);
set(gcf,'color','w');
% set(gca,'YDir','reverse');
end
