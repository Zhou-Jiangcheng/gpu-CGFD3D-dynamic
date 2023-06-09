function [x,y]=coordtrans_ll2xy(lat,lon,ref_lat,ref_lon)

%- set proj, utm zone determined by input reference point
%axesm utm;
fd_utm_zone = utmzone(ref_lat, ref_lon)

%-- set up utm projection
mstruct = defaultm('utm');
mstruct.zone = fd_utm_zone;
mstruct = defaultm(mstruct);

%-- lat/lon to utm
[x,y] = projfwd(mstruct, lat, lon);
