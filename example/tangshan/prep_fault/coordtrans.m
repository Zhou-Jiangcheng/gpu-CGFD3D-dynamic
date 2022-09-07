function [x,y]=coordtrans(lat,lon)

latlon20=[lat,lon]; %latitude first, longitude second

%Set projection mode,utm is the Universal Transverse Mercator (UTM) way define by MATLAT.
axesm utm;

Z=utmzone(latlon20) %this function is choice projection area according to lat and lon
% Z='50R';   % this way is also ok   

%  Z='50T'
setm(gca,'zone',Z);

h = getm(gca);

R=zeros(size(latlon20));

for i=1:length(latlon20)
    
    [x,y]= mfwdtran(h,latlon20(i,1),latlon20(i,2));
    
    R(i,:)=[x;y];
    
end

% x is lon, y is lat 
x=R(:,1);
y=R(:,2);
% x = x - x(281*205+1);
% y = y - y(281*205+1);
%point(1,1) is Local Cartesian coordinates origin
% x = x - x(1);
% y = y - y(1);
end