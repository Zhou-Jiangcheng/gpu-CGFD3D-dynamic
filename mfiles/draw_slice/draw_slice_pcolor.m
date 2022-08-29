clear all;
close all;
clc;
addmypath;
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project1/params.json';
output_dir='../../project1/output';

% which slice to plot
slicedir='z';
sliceid=199;

% which variable and time to plot
varnm='Vx';
ns=100;
ne=800;
nt=50;

% figure control parameters
flag_km     = 1;
flag_emlast = 1;
flag_print  = 0;
scl_daspect =[1 1 1];
clrmp       = 'parula';
taut=0.5;
% ---------------------------------------------------------------------- %

% read parameter file
par=loadjson(parfnm);
ni=par.number_of_total_grid_points_x;
nj=par.number_of_total_grid_points_y;
nk=par.number_of_total_grid_points_z;
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;

% figure plot
hid=figure;
set(hid,'BackingStore','on');

% load data
for nlayer=ns:nt:ne
  % -------------------- slice x ---------------------- %
  if slicedir == 'x'
    [V, X, Y, Z, t] = gather_slice_x(output_dir,nlayer,varnm,sliceid,nproj,nprok);
        
    str_unit='m';
    if flag_km
      X=X/1e3;
      Y=Y/1e3;
      Z=Z/1e3;
      str_unit='km';
    end
        
    pcolor(Y,Z,V);
    xlabel(['Y axis (',str_unit,')']);
    ylabel(['Z axis (',str_unit,')']);
  
  % -------------------- slice y ---------------------- %
  elseif slicedir == 'y'
    [V, X, Y, Z, t] = gather_slice_y(output_dir,nlayer,varnm,sliceid,nproi,nprok);
      
      str_unit='m';
      if flag_km
        X=X/1e3;
        Y=Y/1e3;
        Z=Z/1e3;
        str_unit='km';
      end
      pcolor(X,Z,V);
      xlabel(['X axis (',str_unit,')']);
      ylabel(['Z axis (',str_unit,')']);
  
  % -------------------- slice z ---------------------- %
  else
    [V, X, Y, Z, t] = gather_slice_z(output_dir,nlayer,varnm,sliceid,nproi,nproj);
      
    str_unit='m';
    if flag_km
      X=X/1e3;
      Y=Y/1e3;
      Z=Z/1e3;
      str_unit='km';
    end

    pcolor(X,Y,V);
    xlabel(['X axis (',str_unit,')']);
    ylabel(['Y axis (',str_unit,')']);
      
  end
  
  disp([ '  draw ' num2str(nlayer) 'th time step (t=' num2str(t) ')']);
  
  set(gca,'layer','top');
  set(gcf,'color','white','renderer','painters');

  % axis image
  % shading interp;
  shading flat;
  % colorbar range/scale
  if exist('scl_caxis')
      caxis(scl_caxis);
  end
  % axis daspect
  if exist('scl_daspect')
      daspect(scl_daspect);
  end
  % colormap and colorbar
  if exist('clrmp')
      colormap(clrmp);
  end
  colorbar('vert');
  
  % title
  titlestr=['Snapshot of ' varnm ' at ' ...
            '{\fontsize{12}{\bf ' ...
            num2str((t),'%7.3f') ...
            '}}s'];
  title(titlestr);
  
  drawnow;
  pause(taut);
  
  % save and print figure
  if flag_print==1
      width= 500;
      height=500;
      set(gcf,'paperpositionmode','manual');
      set(gcf,'paperunits','points');
      set(gcf,'papersize',[width,height]);
      set(gcf,'paperposition',[0,0,width,height]);
      fnm_out=[varnm '_ndim_',num2str(nlayer,'%5.5i')];
      print(gcf,[fnm_out '.png'],'-dpng');
  end
  
end
