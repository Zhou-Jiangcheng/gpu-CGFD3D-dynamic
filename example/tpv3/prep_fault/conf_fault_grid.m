% this script is generate fault grid, include strong boundary
% ny,nz is determined by script fault_index_length.m
clc;
clear;
close all;

addmypath;
nj =400; % without ghost points 
nk =200; 
dh = 100; %grid physics length
j1 = 51;
j2 = 351;
k1 = 51;
k2 = 200;

x = zeros(nj, nk);
y = zeros(nj, nk);
z = zeros(nj, nk);

for j = 1:nj
    for k = 1:nk
        x(j,k) = 0;
        y(j,k) = (j-1-nj/2)*dh;
        z(j,k) = (k-nk)*dh;
    end
end

figure
pcolor(y, z, x);
shading flat
colorbar
axis image


disp('calculating metric and base vectors...')
metric = cal_metric(x,y,z);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);
jac = metric.jac;
disp('write output...')
fnm_out = "./fault_coord.nc"
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


