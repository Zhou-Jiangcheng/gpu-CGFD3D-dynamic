clc;
clear all;
close all;
addmypath;
par = loadjson('params.json');
ny = par.NY;
nz = par.NZ;
dh = par.DH; %grid physics length
% get surface points, kinematics coordinate origin is point [min(longitude),min(latitude)
% any point is ok, choose by you
% coordtrans must need Column, lenght >=2
lon_orig = 118.082;
lat_orig = 39.558;
lat1 =[lat_orig;lat_orig+0.1];
lon1 =[lon_orig;lon_orig+0.1];

[x,y]=coordtrans(lat1,lon1);
%source points transform to local coordinate 
[lonsrc,latsrc,v] = textread('tangshan_fault.txt','%f%f%f');
[x1src,y1src]=coordtrans(latsrc,lonsrc);
%faults points transform to local coordinate,minimum longitude points is origin
[min_src,min_p] = min(lonsrc);
%dynamic coordinate coords in kinematics coordinate
xsrc_orig = x1src(min_p) - x(1);
ysrc_orig = y1src(min_p) - y(1);
xsrc = x1src - x(1);
ysrc = y1src - y(1);
%load data xf_inv yf_inv zf_inv
load('x_coords.mat');
load('y_coords.mat');
load('z_coords.mat');

xs = xsrc_orig + xf_inv;
ys = ysrc_orig + yf_inv;
zs = zf_inv;

% m = atan2(ysrc(1)-ysrc(end),xsrc(1)-xsrc(end));
m = atan2(ysrc(end)-ysrc(1),xsrc(end)-xsrc(1));
if(m >= 0 && m <= pi/2)
    angle= 90 - m/pi*180;
elseif(m >= -pi && m < 0 )
    angle= 90 - m/pi*180;
elseif(m>pi/2 && m <= pi)
    angle = 90 - m/pi*180 + 360;
end

%xs,ys,zs max colume is fault Uppermost
for i = 1:length(xs(:,end))-1
if(angle>180)
    kk(i) = atan2(ys(i,end)-ys(i+1,end),xs(i,end)-xs(i+1,end)); %slope
else
    kk(i) = atan2(ys(i+1,end)-ys(i,end),xs(i+1,end)-xs(i,end));
end
if(kk(i) >= 0 && kk(i) <= pi/2)
strike(i) = 90 - kk(i)/pi*180;
elseif(kk(i) >= -pi && kk(i) < 0 )
strike(i) = 90 - kk(i)/pi*180;
elseif(kk(i)>pi/2 && kk(i) <= pi)
strike(i) = 90 - kk(i)/pi*180 + 360;
end

end
strike(length(xs(:,end))) = strike(length(xs(:,end))-1);
dip = 70;
%get fault size
load('t_avg.mat');
[ny_a,nz_a] = size(t_avg);

if 1
figure;
surf(xs, ys, zs); 
% surf(xf_inv, yf_inv, zf_inv); 
shading interp;
title('dongshan-front fault');
xlabel('W-E axis(km)');
ylabel('S-N axis(km)');
zlabel('depth(km)');
axis equal;
view([30, 45]);
set(gca,'FontName','Times New Roman','FontSize',30,'LineWidth',1.5);
set(gcf,'color','w');
end
%%
%load data
load('t_avg.mat');
load('A_avg.mat');
load('t_avg.mat');
load('V_avg.mat');
load('rake_avg.mat');
%%
addmypath;
parfile = 'params.json';
par = loadjson(parfile);
DT = par.DT;
rho = par.rho1;
vs = par.vs1;
TSKP = par.EXPORT_TIME_SKIP;
dt = 3 * DT * TSKP;
dt_in = dt;
nt_in = 200;
miu = rho*vs^2;
nt_max = length(V_avg(:,1,1));
for k =1:nz_a
    for j = 1:ny_a
         t0 = t_avg(j,k);
         nt_start(j,k) = floor(t0/dt);
    end
end
%%
%==============================================================================
%-- write .valsrc file
%==============================================================================
anasrc_file = 'tangshan_source.valsrc';
flag_of_stf = 1; %discrete value
flag_of_cmp = 2; %moment source
mechanism_format = 1; %angle+mu+V+A
flag_loc = 1;     % physical coords
flag_loc_z = 0;   % z axis is physical coords
test_name = 'tangshan_event\n'; %test name
in_num_source = ny_a * nz_a;
fid=fopen(anasrc_file,'w'); % Output file name 

fprintf(fid,test_name); %test name
fprintf(fid,'%g\n',in_num_source);
fprintf(fid,'%g %g %g\n',flag_of_stf,dt_in,nt_in);  %source type, dt, nt
fprintf(fid,'%g %g\n',flag_of_cmp,mechanism_format);
fprintf(fid,'%g %g\n',flag_loc,flag_loc_z);
for k = 1 : nz_a
    for j =1 : ny_a
        fprintf(fid,'%g %g %g\n',xs(j,k), ys(j,k), zs(j,k));
    end
end

for k = 1 : nz_a
  disp(k);
  for j = 1: ny_a
    if(nt_start(j,k)<0 || nt_start(j,k)>nt_max)
      fprintf(fid, '%g\n',9999);
      for i = 1:nt_in
          fprintf(fid,'%g %g %g %g %g %g\n',0,0,0,0,0,0);
      end
    elseif(nt_start(j,k) >= 0)
      fprintf(fid, '%g\n',t_avg(j,k));
      for i = nt_start(j,k)+1:nt_start(j,k)+nt_in
        if(i<=nt_max)
          fprintf(fid,'%g %g %g %g %g %g\n',strike(j),dip,rake_avg(i,j,k), miu,V_avg(i,j,k),A_avg(j,k));
        else
          fprintf(fid,'%g %g %g %g %g %g\n',0,0,0,0,0,0);
        end
      end
    end
  end
end
fclose(fid);






