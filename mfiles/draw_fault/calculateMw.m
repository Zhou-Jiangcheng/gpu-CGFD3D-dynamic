clear all;
close all;
clc;
addmypath;
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project2/test.json'
output_dir='../../project2/output'

par = loadjson(parfnm);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
varnm = 'Slip';
varnm_media = 'mu';
dh = 1; % computer domain is 1
%calculate Mw
id=1;
fault_index = par.fault_x_index;
fault_dir = par.grid_generation_method.fault_plane.fault_geometry_dir;
fault_file = sprintf('%s%s%s%s',fault_dir, "/fault_coord_",num2str(id),".nc"); 
jac  = ncread(fault_file, 'jac');
nt = fault_num_time(output_dir);

[D, t] = gather_fault(output_dir,fault_index(id),nt,varnm,nproj,nprok);
sum(sum(D))
max(max(D))
[mu] = gather_fault_media(output_dir,fault_index(id),varnm_media,nproj,nprok);
A = jac'*dh*dh;
M0 = sum(sum(mu.*D.*A));
Mw = 2*log10(M0)/3.0-6.06;
