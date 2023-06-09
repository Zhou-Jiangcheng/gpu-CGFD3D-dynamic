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
%- settings
%---------------------------------------------------------------

flag_print = 1

parfile    = '../run2/params.json'
output_dir = '../run2/output'
faultfile  = '../prep_fault/fault_coord.nc';
ou_srcfile = 'dyn_rup_latlon.src'

par = loadjson(parfile);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);
dh = par.grid_generation_method.fault_plane.fault_inteval

%-- fault size
ny = j2 - j1 + 1
nz = k2 - k1 + 1

%-- medium, is better to load fd media output nc
Vs  = 3464
Den = 2670

%-- downsampling by how many points
subft_stk_pts = 10;
subft_dip_pts = 10;
subft_time_pts = 1;

%-- time length in .src file per subfault, in second
time_length_of_each_subfault = 5.0

%---------------------------------------------------------------
%- read fault grid
%---------------------------------------------------------------

% get area etc
jac_all   = ncread(faultfile, 'jac');
vec_m_all = ncread(faultfile, 'vec_m'); % strike direction
vec_l_all = ncread(faultfile, 'vec_l'); % dip direction

x_all = ncread(faultfile, 'x'); % x-axis
y_all = ncread(faultfile, 'y'); % 
z_all = ncread(faultfile, 'z'); %

jac   = jac_all(j1:j2,j1:k2);
vec_m = vec_m_all(:,j1:j2,j1:k2);
vec_l = vec_l_all(:,j1:j2,j1:k2);

fault_locx = x_all(j1:j2,j1:k2);
fault_locy = y_all(j1:j2,j1:k2);
fault_locz = z_all(j1:j2,j1:k2);

fault_area = jac*dh*dh;

%---------------------------------------------------------------
%- derive strike and dip
%---------------------------------------------------------------

%-- az is couterclockwise angle from positive x-axis
[az,el,r] = cart2sph(squeeze(vec_m(1,:,:)),squeeze(vec_m(2,:,:)),squeeze(vec_m(3,:,:)));
stk_loc = 90.0 - rad2deg(az);

%-- el is elvation angle from xy-plane
[az,el,r] = cart2sph(squeeze(vec_l(1,:,:)),squeeze(vec_l(2,:,:)),squeeze(vec_l(3,:,:)));
dip_loc = rad2deg(el);

%---------------------------------------------------------------
%- project back to utm and latlon
%---------------------------------------------------------------

% defined as angle start from utm-x to local-x, positive anti-clockwise from utm-x
rot_ang = strike_estimate;

A_stky2utm = [ cosd(rot_ang),  sind(rot_ang);
              -sind(rot_ang),  cosd(rot_ang) ];

%-- rotate
fault_utmx_loc = zeros(size(jac));
fault_utmy_loc = zeros(size(jac));

for k = 1 : nz
for j = 1 : ny
    fault_utmx_loc(j,k) = A_stky2utm(1,:) * [fault_locx(j,k); fault_locy(j,k)];
    fault_utmy_loc(j,k) = A_stky2utm(2,:) * [fault_locx(j,k); fault_locy(j,k)];
end
end

%-- convert to first point as origi
fault_utmx = fault_utmx_loc + ref_utmx0;
fault_utmy = fault_utmy_loc + ref_utmy0;
fault_utmz = fault_locz;

%-- convert to latlon
[fault_lat, fault_lon] = coordtrans_xy2ll(fault_utmx, fault_utmy, ref_lat0, ref_lon0);

%-- convert to depth
fault_dep = - fault_utmz;

%-- local strike/dip to strike/dip
fault_stk = stk_loc + strike_estimate;
fault_dip = dip_loc;

%---------------------------------------------------------------
%- load exported data
%---------------------------------------------------------------

%- dyn_rake, dyn_rate, dyn_tdim, dyn_t0
load dyn_vars.mat;

%---------------------------------------------------------------
%- downsampling
%---------------------------------------------------------------

nt = size(dyn_rake,3);

subft_nj = floor( ny / subft_stk_pts )
subft_nk = floor( nz / subft_dip_pts )
subft_nt = floor( nt / subft_time_pts )

subft_num = subft_nj * subft_nk;

subft_stk_pts_half = round( subft_stk_pts / 2 );
subft_dip_pts_half = round( subft_dip_pts / 2 );

%-- prealloc
subft_rake = zeros([subft_nj,subft_nk,subft_nt]);
subft_rate = zeros([subft_nj,subft_nk,subft_nt]);
subft_area = zeros([subft_nj,subft_nk,subft_nt]);
subft_stk  = zeros([subft_nj,subft_nk,subft_nt]);
subft_dip  = zeros([subft_nj,subft_nk,subft_nt]);

subft_lat  = zeros([subft_nj,subft_nk         ]);
subft_lon  = zeros([subft_nj,subft_nk         ]);
subft_dep  = zeros([subft_nj,subft_nk         ]);
subft_t0   = zeros([subft_nj,subft_nk         ]);

%- method 1: take center value, simplest; time take last sample
for kk = 1 : subft_nk
for jj = 1 : subft_nj
    j = (jj-1) * subft_stk_pts + subft_stk_pts_half;
    k = (kk-1) * subft_dip_pts + subft_dip_pts_half;

    subft_lon(jj,kk) = fault_lon(j,k);
    subft_lat(jj,kk) = fault_lat(j,k);
    subft_dep(jj,kk) = fault_dep(j,k);
    subft_t0 (jj,kk) = dyn_t0 (j,k);

    for tt = 1 : subft_nt
        it = tt * subft_time_pts;
        subft_stk (jj,kk,tt) = fault_stk(j,k);
        subft_dip (jj,kk,tt) = fault_dip(j,k);

        subft_rake(jj,kk,tt) = dyn_rake(j,k,it);
        subft_rate(jj,kk,tt) = dyn_rate(j,k,it);

        %-- area should be summed
        jj1 = (jj-1) * subft_stk_pts + 1;
        kk1 = (kk-1) * subft_dip_pts + 1;
        jj2 = jj1 + subft_stk_pts - 1;
        kk2 = kk1 + subft_dip_pts - 1;
        subft_area(jj,kk,tt) = sum(sum(fault_area(jj1:jj2,kk1:kk2)));
    end

end
end

for tt = 1 : subft_nt
    it = tt * subft_time_pts;
    subft_tval(tt) = dyn_tdim(it);
end
subft_dt = subft_tval(2) - subft_tval(1)

%- method 2: average, need to implement in future

%---------------------------------------------------------------
%- plot
%---------------------------------------------------------------

flag_print = 1

if flag_print == 1

   figure;
   %subft_t0(find(subft_t0<0))=NaN;
   surf(subft_lon, subft_lat, subft_dep, subft_t0); 
   set(gca,'zdir','reverse')

   title('Rutpure front time');
   xlabel('Longitude'); ylabel('Latitude'); zlabel('depth (m)');
   shading flat;
   %alpha(0.8);
   %set(gca,'FontName','Times New Roman','FontSize',30,'LineWidth',1.5);
   %set(gcf,'color','w');
   cval = colormap;
   cval(1,:) = [0.5,0.5,0.5]; % gray color
   colormap(cval);
   caxis([-0.1,max(max(subft_t0))]);
   colorbar
   daspect([1,0.9,85e3])
   view([-5, 30]);
   %camlight('left');  %
   print(gcf,'-dpng','subft_t0.png');

   figure;
   surf(subft_lon, subft_lat, subft_dep, subft_stk(:,:,1)); 
   shading interp;
   title('Subfault strike');
   xlabel('Longitude'); ylabel('Latitude'); zlabel('depth (m)');
   daspect([1,0.9,85e3])
   view([-10, 30]);
   colorbar
   %set(gca,'FontName','Times New Roman','FontSize',30,'LineWidth',1.5);
   %set(gcf,'color','w');
   print(gcf,'-dpng','subft_stk.png');

   figure;
   surf(subft_lon, subft_lat, subft_dep, subft_dip(:,:,1)); 
   shading interp;
   title('Subfault dip');
   xlabel('Longitude'); ylabel('Latitude'); zlabel('depth (m)');
   daspect([1,0.9,85e3])
   view([-10, 30]);
   colorbar
   %set(gca,'FontName','Times New Roman','FontSize',30,'LineWidth',1.5);
   %set(gcf,'color','w');
   print(gcf,'-dpng','subft_stk.png');

   figure;
   surf(subft_lon, subft_lat, subft_dep, subft_area(:,:,1)); 
   shading interp;
   title('Subfault area');
   xlabel('Longitude'); ylabel('Latitude'); zlabel('depth (m)');
   daspect([1,0.9,85e3])
   view([-10, 30]);
   colorbar
   %set(gca,'FontName','Times New Roman','FontSize',30,'LineWidth',1.5);
   %set(gcf,'color','w');
   print(gcf,'-dpng','subft_area.png');

end

%---------------------------------------------------------
%-- push to structure
%---------------------------------------------------------

%-- attention: the first output time sample in C dyn code is not at time intevral
stf_nt = floor( time_length_of_each_subfault / subft_dt )

%-- need to improve in future to incorporate complex structure
src_mu = Den * Vs^2;

%-- count positive t0 only
src_num = numel(find(subft_t0 > - 0.01))

%-- event name
src.evtnm = 'dynsrc';

%-- number of point sources
src.number_of_source = src_num;

%-- stf given by name or discrete values
%   0 analytic stf or 1 discrete values
src.stf_is_discrete = 1;

%for analytical, 2nd value is time length
src.stf_time_length = 0.0;
%for discrete,  2nd value is dt and 3rd is Nt
src.stf_dt = subft_dt;
src.stf_nt = stf_nt;

%-- cmp,  1(force), 2(momoment), 3(force+moment)
src.cmp_fi_mij = 2;

%-- mechanism type: 0 by mij, 1 by angle + mu + D + A
src.mechansim_type = 1;

%-- location by indx or axis: 0 grid index, 1 coords
src.loc_coord_type = 1

%-- 3rd dim is depth or not
src.loc_3dim = 1;

%-- start time of each point source
src.t_start  = subft_t0;

is = 0;
for k = 1 : subft_nk
for j = 1 : subft_nj

    %-- skip negative t0
    if subft_t0(j,k) < 0.0
       continue
    end

    is = is + 1;

    %-- coords
    src.x_coord(is) = subft_lon(j,k);
    src.y_coord(is) = subft_lat(j,k);
    src.z_coord(is) = subft_dep(j,k);

    %-- first t index
    %-- attention: can't use t0/dt 
    %        since the first output time sample in C dyn code is not at time intevral
    if subft_tval(end) < subft_t0(j,k)
      sf_it1 = subft_nt + 1;
    else
      sf_it1 = find(subft_tval >= subft_t0(j,k),1);
    end

    %-- values
    for it = 1 : src.stf_nt

      sf_it = sf_it1 + it - 1;

      src.strike(it, is) = subft_stk(j,k);
      src.dip   (it, is) = subft_dip(j,k);
      src.mu    (it, is) = src_mu;
      src.A     (it, is) = subft_area(j,k);

      if sf_it > subft_nt
        src.rake  (it, is) = 0;
        src.D     (it, is) = 0;
      else
        src.rake  (it, is) = subft_rake(j,k,sf_it);
        src.D     (it, is) = subft_rate(j,k,sf_it);
      end

    end

end
end

%------------------------------------------------------------------------------
%-- export
%------------------------------------------------------------------------------

src_export(ou_srcfile, src);

