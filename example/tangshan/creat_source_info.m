clc;
clear;
close all;

addmypath;
parfile = 'params.json';
par = loadjson( parfile);
DT = par.DT;
dh = par.DH;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;
rho = par.rho1;
vs= par.vs1;
Faultgrid = par.Fault_grid;
j1 = Faultgrid(1);
j2 = Faultgrid(2);
k1 = Faultgrid(3);
k2 = Faultgrid(4);
ny_a = j2-j1+1;
nz_a = k2-k1+1;

NT = floor(TMAX/(DT*TSKP));
%time triple Downsampling
%time length
nt = floor(NT/3);
for i = 1 : nt
  
  it = 3*i;
  disp(it);
  
  Vs1 = gather_snap(parfile,'Vs1',it);
  Vs2 = gather_snap(parfile,'Vs2',it);
  vs1 = Vs1(j1:j2,k1:k2);
  vs2 = Vs2(j1:j2,k1:k2);
  V(:,:) = sqrt(vs1.^2+vs2.^2);
  rake(:,:) = atan2(vs2(:,:),(vs1(:,:)))./pi*180;
  %space Downsampling 
  %get average rake and Vs (1*1 --> 1) 
  %not use space Downsampling there
  rake_avg(i,:,:) = rake;
  V_avg(i,:,:) = V;
end
save('rake_avg.mat','rake_avg');
save('V_avg.mat','V_avg');
%%
% get area
faultfile   = 'fault_coord.nc';
jac  = ncread(faultfile, 'jac');
Jac = jac(j1:j2,k1:k2);
miu = rho*vs^2;
A = Jac*dh*dh;
A_avg = A;
save('A_avg.mat','A_avg');
%%
% get avrage init t0
% (1*1 --> 1)
t = gather_snap(parfile,'init_t0');
t1 = t(j1:j2 , k1:k2);
t_avg = t1;
save('t_avg.mat','t_avg');

        
