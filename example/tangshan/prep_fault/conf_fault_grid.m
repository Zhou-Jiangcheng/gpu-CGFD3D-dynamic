% this script is generate fault grid, include strong boundary
% ny,nz is readed by json, determined by script fault_index_length.m
clc;
clear;
close all;

addmypath;

ny = 960;
nz = 300;
dh = 90; %grid physics length
j1 = 101;
j2 = 896;
k1 = 78;
k2 = 300;
%read faults points and project
[lonsrc,latsrc,v] = textread('tangshan_fault.txt','%f%f%d');
[x1src,y1src]=coordtrans(latsrc,lonsrc);
%faults points transform to local coordinate,minimum longitude points is origin
[min_src,min_p] = min(lonsrc);
xsrc = x1src - x1src(min_p);
ysrc = y1src - y1src(min_p);

%Calculate the angle of the first fault for rotating the coordinate system
%NOTE !!!!
%must follow arcgis sequence of exported fault points
%then use atan2(ysrc(1)-ysrc(end),xsrc(1)-xsrc(end)) or 
%         atan2(ysrc(end)-ysrc(1),xsrc(end)-xsrc(1))
% to calculate the strike. use which one depend on real strike

% m = atan2(ysrc(end)-ysrc(1),xsrc(end)-xsrc(1));
m = atan2(ysrc(1)-ysrc(end),xsrc(1)-xsrc(end));
if(m >= 0 && m <= pi/2)
    angle= 90 - m/pi*180;
elseif(m >= -pi && m < 0 )
    angle= 90 - m/pi*180;
elseif(m>pi/2 && m <= pi)
    angle = 90 - m/pi*180 + 360;
end
%NOTE if angle > 180, rotation angle is minus 180
if(angle>180)
rota = (angle-180)/180*pi;
end
if(angle<180)
rota = angle/180*pi;
end

%rotation matrix,clockwise rotate.
A = [cos(rota),-sin(rota);
    sin(rota),cos(rota)];
%Rotate the coordinate system. The purpose is to reduce the simulation area
for i = 1:length(xsrc)
    x2src(i) = A(1,:) * [xsrc(i);ysrc(i)];
    y2src(i) = A(2,:) * [xsrc(i);ysrc(i)];
end

% interpolation, surface interval length is 100m
xi = [];yi=[];
for i = 1:length(xsrc)-1
    len(i)= sqrt((y2src(i+1)-y2src(i))^2+(x2src(i+1)-x2src(i))^2);
    np(i) = round(len(i)/dh);
    xx = linspace(x2src(i),x2src(i+1),np(i));
    xx = xx(1:np(i)-1);
    yy = linspace(y2src(i),y2src(i+1),np(i));
    yy = yy(1:np(i)-1);
    xi{end+1} = xx;
    yi{end+1} = yy;
end
xi = cell2mat(xi);
yi =  cell2mat(yi);

%!!!!!!!!NOTE!!!!!!!!
%maybe need fliplr or not
%depend on conf_fault_stress.m
if(angle < 180)
    %xi = fliplr(xi);
    %yi = fliplr(yi);
end
if(angle > 180)
    %xi = fliplr(xi);
    %yi = fliplr(yi);
end
%smooth fault surface
xi_s = smoothdata(xi,'movmean');
figure(2)
plot(xi,yi,'r');
hold on;
plot(xi_s,yi,'b');
axis equal
%use smooth x coords
xi = xi_s;

% % b is fault strike direction index number
b = length(xi)
%left strong boundary, 100 grid length
for i = 1:100
    x(i) = xi(1)+(xi(2)-xi(1))*(i-101);
    y(i) = yi(1)+(yi(2)-yi(1))*(i-101);
end
%fault zone
for i = 101:100+b
    x(i) = xi(i-100);
    y(i) = yi(i-100);
end
% right strong boundary
for i = b+101:ny
    x(i) = xi(b)+(xi(b)-xi(b-1))*(i-b-100);
    y(i) = yi(b)+(yi(b)-yi(b-1))*(i-b-100);
end

dip = 85;
% fault depth is 0-20km, interval length is 90m
% nz is 300*90m, include strong boundary
zi = -dh*(nz-1):dh:0;

% Note direction of the Z axis is upward, surface is 0.
%if strike < 180 , dip to right
% so we take minus(-)
%if strike > 180 , dip to left
% so we take add(+)
if(angle < 180)
    for k =k2+1:nz
        zx(k) = 0; 
    end
    for k =1:k2
        zx(k) = -zi(k+nz-k2)/tand(dip); 
    end
end
if(angle > 180)
    for k =k2+1:nz
        zx(k) = 0; 
    end
    for k =1:k2
        zx(k) = zi(k+nz-k2)/tand(dip); 
    end
end
for k =1:nz
    for j =1:ny
      xf(j,k) = x(j) + zx(k)*cosd(0);
      yf(j,k) = y(j) + zx(k)*sind(0);
      zf(j,k) = zi(k);
    end
end

% inverse Rotate the coordinate system
% north is y axis
if angle > 180
    angle_inv = -(angle - 180);
end
if angle < 180
    angle_inv = -angle;
end
rota_inv = angle_inv/180*pi;
B = [cos(rota_inv),-sin(rota_inv);
    sin(rota_inv),cos(rota_inv)];
for k =1:k2-k1+1
    for j =1:j2-j1+1
      xy_inv = B * [xf(j+j1-1,k+k1-1);yf(j+j1-1,k+k1-1)];
      xf_inv(j,k) =  xy_inv(1);
      yf_inv(j,k) =  xy_inv(2); 
      zf_inv(j,k) =  zi(k+k1-1);
    end
end

save('x_coords.mat','xf_inv');
save('y_coords.mat','yf_inv');
save('z_coords.mat','zf_inv');



disp('calculating metric and base vectors...')
metric = cal_metric(xf,yf,zf,dh);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);
jac = metric.jac;
disp('write output...')
fnm_out = "./fault_coord_1.nc"
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
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
netcdf.putVar(ncid,varid(1),xf);
netcdf.putVar(ncid,varid(2),yf);
netcdf.putVar(ncid,varid(3),zf);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.putVar(ncid,varid(7),jac);
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


