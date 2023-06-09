function [lat,lon]=coordtrans_xy2ll(utmx,utmy,ref_lat,ref_lon)

%- set proj, utm zone determined by input reference point
%axesm utm;
fd_utm_zone = utmzone(ref_lat, ref_lon)

%-- set up utm projection
mstruct = defaultm('utm');
mstruct.zone = fd_utm_zone;
mstruct = defaultm(mstruct);

%-- utm to lat/lon
[lat, lon] = projinv(mstruct, utmx, utmy);

