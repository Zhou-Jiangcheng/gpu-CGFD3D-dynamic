clc;
clear all;
close all;
addmypath;
parfile = 'params.json';
par = loadjson( parfile);
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;
DT = par.DT;
dh = par.DH;
NT = floor(TMAX/(DT*TSKP));
rho = par.rho1;
vs= par.vs1;
%calculate Mw
faultfile   = 'fault_coord.nc';
jac  = ncread(faultfile, 'jac');
D = gather_snap(parfile,'Us0',NT);
miu = rho*vs^2;
A = jac*dh*dh;
M0 = miu*sum(sum(D.*A));
Mw = 2*log10(M0)/3.0-6.06;