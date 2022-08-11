% this script is generate fault grid, include strong boundary
% ny,nz is readed by json, determined by script fault_index_length.m
clc;
clear;
close all;

addmypath;
parfnm='../project/params.json'
output_dir='../project/output'
par = loadjson(parfnm);
ny = par.number_of_total_grid_points_y;
nz = par.number_of_total_grid_points_z;
dh = 100; %grid physics length

for i = 1:ny
    x(i) = 0;
    y(i) = 0 + (i-1)*dh;
end
z = -dh*(nz-1):dh:0;
for k =1:nz
    for j =1:ny
      xf(j,k) = x(j);
      yf(j,k) = y(j);
      zf(j,k) = z(k);
    end
end


disp('calculating metric and base vectors...')
disp('write output...')
fnm_out = [pwd,'/','fault_grid.nc'];
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);

netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),xf);
netcdf.putVar(ncid,varid(2),yf);
netcdf.putVar(ncid,varid(3),zf);

netcdf.close(ncid);

%%
if 1
figure;
surf(xf(j1:j2,k1:k2)/1e3, yf(j1:j2,k1:k2)/1e3, zf(j1:j2,k1:k2)/1e3); 
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


