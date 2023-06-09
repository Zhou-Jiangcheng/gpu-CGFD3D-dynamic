clc;
clear all;
close all;

set_mpath

%---------------------------------------------------------------
%- load conf grid parameters
%---------------------------------------------------------------

dyn_par_set
%- read strike to get coord origin
[stk_lon, stk_lat, stk_elv] = textread(fault_strike_file, '%f%f%f');
%  use first point as reference point to determine utm zone and local coord
ref_lat0 = stk_lat(1); ref_lon0 = stk_lon(1);
[stk_utmx, stk_utmy]=coordtrans_ll2xy(stk_lat, stk_lon, ref_lat0, ref_lon0);
ref_utmx0 = stk_utmx(1); ref_utmy0 = stk_utmy(1);

%---------------------------------------------------------------
%- import, convert and export
%---------------------------------------------------------------

ll_srcfile = 'dyn_rup_latlon.src'
xy_srcfile = 'dyn_rup_cart.src'

src = src_import(ll_srcfile);

ft_lon = src.x_coord;
ft_lat = src.y_coord;

[ft_utmx, ft_utmy] = coordtrans_ll2xy(ft_lat, ft_lon, ref_lat0, ref_lon0);

ft_x = ft_utmx - ref_utmx0;
ft_y = ft_utmy - ref_utmy0;

src.x_coord = ft_x;
src.y_coord = ft_y;

src_export(xy_srcfile, src);

