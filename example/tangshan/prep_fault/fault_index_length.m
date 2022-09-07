%calculate fault index length
clc;
clear;
addmypath;
dh = 100; %grid physics length

%read faults points and project
[lonsrc,latsrc,v] = textread('tangshan_fault.txt','%f%f%d');
[x1src,y1src]=coordtrans(latsrc,lonsrc);
%faults points transform to local coordinate,first points is origin
xsrc = x1src - x1src(1);
ysrc = y1src - y1src(1);
%Calculate the angle of the first fault for rotating the coordinate system

xi = [];yi=[];
for i = 1:length(xsrc)-1
len(i)= sqrt((ysrc(i+1)-ysrc(i))^2+(xsrc(i+1)-xsrc(i))^2);
np(i) = round(len(i)/dh);
xx = linspace(xsrc(i),xsrc(i+1),np(i));
xx = xx(1:np(i)-1);
yy = linspace(ysrc(i),ysrc(i+1),np(i));
yy = yy(1:np(i)-1);
xi{end+1} = xx;
yi{end+1} = yy;
end
xi = cell2mat(xi);
yi =  cell2mat(yi);
xi = xi';
yi = yi';

% a is fault dip direction index number
% b is fault strike direction index number
fault_dip_length = 20000;
a = fault_dip_length / dh + 1
b = length(xi)
