clear all;
close all;
clc;
addmypath;
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/params.json'
output_dir='../../project/output'
% parfnm='../../project2/params.json'
% output_dir='../../project2/output'
par = loadjson(parfnm);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
varnm = 'slip';
varnm_media = 'mu';
dh = 1; % computer domain is 1
%calculate Mw
faultfile = par.grid_generation_method.fault_plane.fault_geometry_file;
jac  = ncread(faultfile, 'jac');
nt = fault_num_time(output_dir);

[D, t] = gather_fault(output_dir,nt,varnm,nproj,nprok);
sum(sum(D))
max(max(D))
[mu] = gather_fault_media(output_dir,varnm_media,nproj,nprok);
A = jac'*dh*dh;
M0 = sum(sum(mu.*D.*A));
Mw = 2*log10(M0)/3.0-6.06;