% this script is to generate fault grid

clc;
clear;
close all;

flag_print = 0

%set_mpath

%---------------------------------------------------------------
%-- read in setting
%---------------------------------------------------------------

dyn_par_set

%---------------------------------------------------------------
%- read and interp strike 
%---------------------------------------------------------------

%-- read in fault strike, stk_elv is omitted now, should consider in future
[stk_lon, stk_lat, stk_elv] = textread(fault_strike_file, '%f%f%f');

%-- fitted strike, angle clockwise from north from first point to end point
%     should use polyfit in the future to best fit the line
%     used for local y-axis, thus x-axis is positive to hanging wall
strike_fit = azimuth(stk_lat(1), stk_lon(1), stk_lat(end), stk_lon(end))

strike_diff = strike_fit - strike_estimate
if abs(strike_diff) > 40
  error('input esimated strike and calculated strike_fit differ')
end

%-- project to UTM
%  use first point as reference point to determine utm zone and local coord
ref_lat0 = stk_lat(1); ref_lon0 = stk_lon(1);
[stk_utmx, stk_utmy]=coordtrans_ll2xy(stk_lat, stk_lon, ref_lat0, ref_lon0);
ref_utmx0 = stk_utmx(1); ref_utmy0 = stk_utmy(1);

if flag_print == 1
   figure
   plot(stk_lon, stk_lat, 'v-')
   print(gcf,'-dpng', 'Fault_strike_latlon_input.png');

   figure
   plot(stk_utmx, stk_utmy, 'v-')
   print(gcf,'-dpng', 'Fault_strike_utmxy_input.png');
end

num_seg = length(stk_lon);

%%-- interp
npt_stk_interp = 1; %-- current point index and total points
for n = 1 : num_seg - 1
    seg_len = sqrt( (stk_utmx(n)-stk_utmx(n+1))^2 + (stk_utmy(n)-stk_utmy(n+1))^2 );
    seg_npt = round( seg_len / DLEN ) + 1;
    seg_xvec = linspace(stk_utmx(n), stk_utmx(n+1), seg_npt);
    seg_yvec = linspace(stk_utmy(n), stk_utmy(n+1), seg_npt);
    stk_intp_x_interp(npt_stk_interp : npt_stk_interp + seg_npt - 1) = seg_xvec;
    stk_intp_y_interp(npt_stk_interp : npt_stk_interp + seg_npt - 1) = seg_yvec;
    npt_stk_interp = npt_stk_interp + seg_npt - 1;
end

%-- use spline fitting
dx = DLEN * cosd(90.0-strike_estimate)
dy = DLEN * sind(90.0-strike_estimate)

%-- avoid strike = 0 or 90 problem
if abs(dx) > abs(dy)
  npt_stk_spline = round( (stk_utmx(end)-stk_utmx(1)) / dx );
  stk_intp_x_spline = [0:npt_stk_spline-1] * dx + stk_utmx(1);
  spl = spline(stk_utmx, stk_utmy);
  stk_intp_y_spline = ppval(spl,stk_intp_x_spline);
else
  npt_stk_spline = round( (stk_utmy(end)-stk_utmy(1)) / dy );
  stk_intp_y_spline = [0:npt_stk_spline-1] * dy + stk_utmy(1);
  spl = spline(stk_utmy, stk_utmx);
  stk_intp_x_spline = ppval(spl,stk_intp_y_spline);
end

if flag_print == 1
   figure
   plot(stk_intp_x_interp, stk_intp_y_interp, 'k-')
   hold on
   plot(stk_intp_x_spline, stk_intp_y_spline, 'r-')
   plot(stk_utmx, stk_utmy, 'b*')
   legend('linear interp','spline','input points');
   title('Refined strike');
   print(gcf,'-dpng', 'Fault_strike_refined.png');
end

%use smooth x coords
if is_smooth_stirke == 1
   stk_intp_x = stk_intp_x_spline;
   stk_intp_y = stk_intp_y_spline;
   npt_stk    = npt_stk_spline;
else
   stk_intp_x = stk_intp_x_interp;
   stk_intp_y = stk_intp_y_interp;
   npt_stk    = npt_stk_interp;
end

%-- extend points along strike for rutpure calculation
dx = stk_intp_x(1) - stk_intp_x(2);
dy = stk_intp_y(1) - stk_intp_y(2);
xvec1 = [NPAD_LEN_START : -1 : 1] * dx + stk_intp_x(1);
yvec1 = [NPAD_LEN_START : -1 : 1] * dy + stk_intp_y(1);

dx = stk_intp_x(end) - stk_intp_x(end-1);
dy = stk_intp_y(end) - stk_intp_y(end-1);
xvec2 = [1 : NPAD_LEN_END] * dx + stk_intp_x(end);
yvec2 = [1 : NPAD_LEN_END] * dy + stk_intp_y(end);

ft_top_x = [xvec1, stk_intp_x, xvec2];
ft_top_y = [yvec1, stk_intp_y, yvec2];

%-- number of points along strike
ny = length(ft_top_x);

%-- strike angle of each elem

% strike is clockwise from north (or y-axis of UTM)
%  atan2d anti-clockwise from x-axis (y,x), swap y,x for atan2d is just strike (x,y)
for n = 1 : ny-1
    dx = ft_top_x(n+1) - ft_top_x(n);
    dy = ft_top_y(n+1) - ft_top_y(n);
    ft_top_stk(n) = atan2d(dx, dy); 
end
ft_top_stk(ny) = ft_top_stk(ny-1);

% atan2d [-180,180], covert minus to [180,360]
indx = find(ft_top_stk < 0.0);
ft_top_stk(indx) = 360.0 + ft_top_stk(indx);

if flag_print == 1
   figure;
   plot(ft_top_x, ft_top_y, 'r-')
   hold on
   plot(stk_intp_x, stk_intp_y, 'b*')
   legend('padding along strike','fault region');
   title('Point padding along Strike');
   print(gcf,'-dpng', 'Fault_strike_after_padding.png');

   figure;
   plot(ft_top_x, ft_top_stk, 'r-')
   hold on
   title('Strike angle of each point');
   print(gcf,'-dpng', 'Fault_strike_of_each_point.png');
end

%---------------------------------------------------------------
%- read dip 
%---------------------------------------------------------------

%-- read in fault dip
[dip_ang, dip_len] = textread(fault_dip_file, '%f%f');
% convert km to m, should input in m
%dip_len = dip_len * 1e3;

%-- fault depth
dip_dep  = cumsum( dip_len .* sind(dip_ang) );

%-- the fault is not stright line, so the strike is varialble
%--  borrow from Prof. Dongli Zhang
dip_len_on_xy = dip_len .* cosd(dip_ang);

num_dip_seg = length(dip_ang);

%---------------------------------------------------------------
%- built fault x,y,z coords for dip in segment format
%---------------------------------------------------------------

ft_dipseg_x = zeros([ny, num_dip_seg+1]);
ft_dipseg_y = zeros([ny, num_dip_seg+1]);
ft_dipseg_d = zeros([ny, num_dip_seg+1]);

for j = 1 : ny

    %-- dx,dy along dip
    %stk_for_dip_ext = ft_top_stk(j); % will create jump space at bottom
    stk_for_dip_ext = strike_estimate; % use average strike

    dip_dietx = dip_len_on_xy * sind(stk_for_dip_ext + 90.0);
    dip_diety = dip_len_on_xy * cosd(stk_for_dip_ext + 90.0);

    %-- x
    ft_dipseg_x(j, 1) = ft_top_x(j);
    for k = 2 : num_dip_seg+1
       ft_dipseg_x(j, k) = ft_dipseg_x(j, k-1) + dip_dietx(k-1);
    end

    %-- y
    ft_dipseg_y(j, 1) = ft_top_y(j);
    for k = 2 : num_dip_seg+1
       ft_dipseg_y(j, k) = ft_dipseg_y(j, k-1) + dip_diety(k-1);
    end

    %-- depth
    ft_dipseg_d(j, 1) = 0;
    ft_dipseg_d(j, 2:end) = dip_dep;

end

if flag_print == 1

   %-- utm coord
   figure;
   surf(ft_dipseg_x, ft_dipseg_y, ft_dipseg_d);
   set(gca,'zdir','reverse');
   colorbar

   xlabel('UTM x-axis (m)');
   ylabel('UTM y-axis (m)');
   zlabel('depth (m)');
   title('Fault plane after refine along strike');

   daspect([2 2 1]);
     set(gca,'box','on');
     view(-20,25);
     colormap(gca,'winter');  
      axis tight;
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_strike_interp.png');

   %-- latlon coord
   figure;
   [ft_dipseg_lat, ft_dipseg_lon] = coordtrans_xy2ll(ft_dipseg_x, ft_dipseg_y, ref_lat0, ref_lon0);
   surf(ft_dipseg_lon, ft_dipseg_lat, ft_dipseg_d);
   set(gca,'zdir','reverse');
   colorbar

   xlabel('Latitude');
   ylabel('Longitude');
   zlabel('depth (m)');
   title('Fault plane after refine along strike');

   set(gca,'box','on');
   view(-20,25);
   colormap(gca,'winter');  
    axis tight;
    grid on;
   camlight('left');  %
   alpha(0.8);
   shading flat

   print(gcf,'-dpng','Fault_plane_strike_interp_latlon.png');

end

%---------------------------------------------------------------
%- built fault x,y,z coords with dip interpolation
%---------------------------------------------------------------

%-- interp along dip
%--   right now use a constant dz for simple implementation (it should be file if dip is not too small)
npt_dip = ceil( dip_dep(end) / DWID ) + 1

%- pad points
nz = npt_dip + NPAD_DIP


fault_x = zeros([ny, nz]);
fault_y = zeros([ny, nz]);
fault_d = zeros([ny, nz]);

for j = 1 : ny

    %-- dx,dy along dip
    %stk_for_dip_ext = ft_top_stk(i); % will create jump space at bottom
    stk_for_dip_ext = strike_estimate; % use average strike

    n1_dip = 1;

    %-- top layer
    fault_x(j, 1) = ft_top_x(j);
    fault_y(j, 1) = ft_top_y(j);
    fault_d(j, 1) = 0.0;

    for k = 2 : nz

        dep = (k-1) * DWID;

        %-- locate which dip segment
        for n = n1_dip : num_dip_seg
            if dip_dep(n) >= dep 
               n1_dip = n;
               break;
            end
        end

        %-- dx,dy due to dip
        dip_elem_xy = DWID * cotd( dip_ang(n1_dip) );
        dip_elem_dietx = dip_elem_xy * sind(stk_for_dip_ext + 90.0);
        dip_elem_diety = dip_elem_xy * cosd(stk_for_dip_ext + 90.0);

        %-- x
        fault_x(j, k) = fault_x(j, k-1) + dip_elem_dietx;

        %-- y
        fault_y(j, k) = fault_y(j, k-1) + dip_elem_diety;

        %-- depth
        fault_d(j, k) = dep;

    end % k
end %j

if flag_print == 1

   %-- utm coord
   figure;
   surf(fault_x, fault_y, fault_d);
   set(gca,'zdir','reverse');
   colorbar

   xlabel('UTM x-axis (m)');
   ylabel('UTM y-axis (m)');
   zlabel('depth (m)');
   title('Grid of fault plane');

   daspect([2 2 1]);
     set(gca,'box','on');
     view(-20,25);
     colormap(gca,'winter');  
      axis tight;
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_grid.png');

   %-- latlon coord
   figure;
   [fault_lat, fault_lon] = coordtrans_xy2ll(fault_x, fault_y, ref_lat0, ref_lon0);
   surf(fault_lon, fault_lat, fault_d);
   set(gca,'zdir','reverse');
   colorbar

   xlabel('Latitude');
   ylabel('Longitude');
   zlabel('depth (m)');
   title('Grid of fault plane');

   set(gca,'box','on');
   view(-20,25);
   colormap(gca,'winter');  
    axis tight;
    grid on;
   camlight('left');  %
   alpha(0.8);
   shading flat

   print(gcf,'-dpng','Fault_plane_grid_latlon.png');

end

%---------------------------------------------------------------
%- conver to z-axis positive upward
%---------------------------------------------------------------

fault_z = 0.0 - fault_d;

fd_fault_x = flipdim(fault_x,2);
fd_fault_y = flipdim(fault_y,2);
fd_fault_z = flipdim(fault_z,2);

if flag_print == 1

   %-- utm coord
   figure;
   surf(fd_fault_x, fd_fault_y, fd_fault_z);
   colorbar

   xlabel('UTM x-axis (m)');
   ylabel('UTM y-axis (m)');
   zlabel('z-axis (m)');
   title('Grid of fault plane');

   daspect([2 2 1]);
     set(gca,'box','on');
     view(-20,25);
     colormap(gca,'winter');  
      axis tight;
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_grid_fdz.png');
end

%---------------------------------------------------------------
%- convert to strike dir as y-axis pos
%---------------------------------------------------------------

% defined as angle start from utm-x to local-x, positive anti-clockwise from utm-x
rot_ang = - strike_estimate;

A_utm2stky = [ cosd(rot_ang),  sind(rot_ang);
              -sind(rot_ang),  cosd(rot_ang) ];

%-- convert to first point as origi
fault_relative_x = fd_fault_x - ref_utmx0;
fault_relative_y = fd_fault_y - ref_utmy0;

%-- rotate
stky_fault_x = zeros(size(fault_relative_x));
stky_fault_y = zeros(size(fault_relative_x));
stky_fault_z = zeros(size(fault_relative_x));

for k = 1 : nz
for j = 1 : ny
    stky_fault_x(j,k) = A_utm2stky(1,:) * [fault_relative_x(j,k); fault_relative_y(j,k)];
    stky_fault_y(j,k) = A_utm2stky(2,:) * [fault_relative_x(j,k); fault_relative_y(j,k)];
    stky_fault_z(j,k) = fd_fault_z(j,k);
end
end

stky_xmin = min(min(stky_fault_x))
stky_ymin = min(min(stky_fault_y))
stky_zmin = min(min(stky_fault_z))
stky_xmax = max(max(stky_fault_x))
stky_ymax = max(max(stky_fault_y))
stky_zmax = max(max(stky_fault_z))

if flag_print == 1

   %-- local coord
   figure;
   surf(stky_fault_x, stky_fault_y, stky_fault_z);
   colorbar

   xlabel('local x-axis (m)');
   ylabel('local y-axis (m)');
   zlabel('z-axis (m)');
   title('Grid of fault plane');

   %stky_ylen = stky_ymax-stky_ymin;
   %x1 = stky_xmin - stky_ylen / 5;
   %x2 = stky_xmax + stky_ylen / 5;
   x1 = min(-10e3, stky_xmin-5e3);
   x2 = max(20e3,  stky_xmax+5e3);

   daspect([2 2 1]);
   axis tight;
   xlim([x1,x2]);
     set(gca,'box','on');
     view(40,30);
     colormap(gca,'winter');  
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_grid_local.png');
end

%---------------------------------------------------------------
%- calculate normal direction
%---------------------------------------------------------------

disp('calculating metric and base vectors...')
%-- DX will be used to expand grid in C code
metric = cal_metric(stky_fault_x,stky_fault_y,stky_fault_z,DX);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);
jac = metric.jac;

flag_print = 1

if flag_print == 1

   figure;

   skip_j = 20;
   skip_k = 20;
   %quiver3(stky_fault_x, stky_fault_y, stky_fault_z, ...
   %           squeeze(vec_n(1,:,:)), ...
   %           squeeze(vec_n(2,:,:)), ...
   %           squeeze(vec_n(3,:,:)));
   quiver3(  stky_fault_x(1:skip_j:end,1:skip_k:end), ...
             stky_fault_y(1:skip_j:end,1:skip_k:end), ...
             stky_fault_z(1:skip_j:end,1:skip_k:end), ...
          squeeze(vec_n(1,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_n(2,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_n(3,1:skip_j:end,1:skip_k:end)));

   colorbar

   xlabel('local x-axis (m)');
   ylabel('local y-axis (m)');
   zlabel('z-axis (m)');
   title('vec n on fault');
   x1 = min(-10e3, stky_xmin-5e3);
   x2 = max(20e3,  stky_xmax+5e3);
   daspect([2 2 1]);
   axis tight;
   xlim([x1,x2]);
     set(gca,'box','on');
     view(40,30);
    % colormap(gca,'winter');  
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_vecn.png');

   figure;
   skip_j = 80;
   skip_k = 20;
   quiver3(  stky_fault_x(1:skip_j:end,1:skip_k:end), ...
             stky_fault_y(1:skip_j:end,1:skip_k:end), ...
             stky_fault_z(1:skip_j:end,1:skip_k:end), ...
          squeeze(vec_m(1,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_m(2,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_m(3,1:skip_j:end,1:skip_k:end)));
   colorbar

   xlabel('local x-axis (m)');
   ylabel('local y-axis (m)');
   zlabel('z-axis (m)');
   title('vec m on fault');
   x1 = min(-10e3, stky_xmin-5e3);
   x2 = max(20e3,  stky_xmax+5e3);
   daspect([2 2 1]);
   axis tight;
   xlim([x1,x2]);
     set(gca,'box','on');
     view(40,30);
     %colormap(gca,'winter');  
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_vecm.png');

   figure;
   skip_j = 40;
   skip_k = 80;
   quiver3(  stky_fault_x(1:skip_j:end,1:skip_k:end), ...
             stky_fault_y(1:skip_j:end,1:skip_k:end), ...
             stky_fault_z(1:skip_j:end,1:skip_k:end), ...
          squeeze(vec_l(1,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_l(2,1:skip_j:end,1:skip_k:end)), ...
          squeeze(vec_l(3,1:skip_j:end,1:skip_k:end)));
   colorbar

   xlabel('local x-axis (m)');
   ylabel('local y-axis (m)');
   zlabel('z-axis (m)');
   title('vec l on fault');
   x1 = min(-10e3, stky_xmin-5e3);
   x2 = max(20e3,  stky_xmax+5e3);
   daspect([2 2 1]);
   axis tight;
   xlim([x1,x2]);
     set(gca,'box','on');
     view(40,30);
     %colormap(gca,'winter');  
      grid on;
     camlight('left');  %
     alpha(0.8);
     shading flat

   print(gcf,'-dpng','Fault_plane_vecl.png');
end

%---------------------------------------------------------------
%- save to nc file
%---------------------------------------------------------------

disp(['write to ', fnm_fault_grid]);

ncid = netcdf.create(fnm_fault_grid, 'CLOBBER');
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
%-- dim order in nc is diff with matlab 
netcdf.putVar(ncid,varid(1),stky_fault_x);
netcdf.putVar(ncid,varid(2),stky_fault_y);
netcdf.putVar(ncid,varid(3),stky_fault_z);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.putVar(ncid,varid(7),jac);
netcdf.close(ncid);

disp('Finish writting output')

%---------------------------------------------------------------
%- write log file
%---------------------------------------------------------------
fid = fopen(fnm_log_grid,'w')

fprintf(fid,'ny = %d\n', ny);
fprintf(fid,'nz = %d\n', nz);
fprintf(fid,'npt_stk = %d\n', npt_stk);
fprintf(fid,'npt_dip = %d\n', npt_dip);
fprintf(fid,'ref_lat0 = %g\n', ref_lat0);
fprintf(fid,'ref_lon0 = %g\n', ref_lon0);
fprintf(fid,'ref_utmx0 = %g\n', ref_utmx0);
fprintf(fid,'ref_utmy0 = %g\n', ref_utmy0);

fclose(fid)

