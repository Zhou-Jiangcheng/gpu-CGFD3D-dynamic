function metricinfo = locate_metric(parfnm,varargin)

% locate metric index in mpi threads
% Author:   Yuanhang Huo
% Email:    yhhuo@mail.ustc.edu.cn
% Date:     2021.06.06

gtstart  = 1;
gtcount  = -1;
gtstride = 1;

%-- flags --
n=1;
while n<=nargin-1
    
    if numel(varargin{n})==1 | ~isnumeric(varargin{n})
        switch varargin{n}
            case 'start'
                gsubs=varargin{n+1}; n=n+1;
                if length(gsubs)==4
                    gtstart=gsubs(4);gsubs=gsubs(1:3);
                end
            case 'count'
                gsubc=varargin{n+1}; n=n+1;
                if length(gsubc)==4
                    gtcount=gsubc(4);gsubc=gsubc(1:3);
                end
            case 'stride'
                gsubt=varargin{n+1}; n=n+1;
                if length(gsubt)==4
                    gtstride=gsubt(4);gsubt=gsubt(1:3);
                end
            case 'metricdir'
                metric_dir=varargin{n+1}; n=n+1;
        end
    end
    
    n=n+1;
    
end

% check parameter file exist
if ~ exist(parfnm,'file')
    error([mfilename ': file ' parfnm ' does not exist']);
end

% read parameters file
par=loadjson(parfnm);
metric_subs=[1, 1, 1];
metric_subc=[-1,-1,-1];
metric_subt=[1, 1, 1];
snap_tinv=1;
ngijk=[par.number_of_total_grid_points_x,...
       par.number_of_total_grid_points_y,...
       par.number_of_total_grid_points_z];

% reset count=-1 to total number
indx=find(metric_subc==-1);
metric_subc(indx)=fix((ngijk(indx)-metric_subs(indx))./metric_subt(indx))+1;
snap_sube=metric_subs+(metric_subc-1).*metric_subt;

indx=find(gsubc==-1);
gsubc(indx)=fix((metric_subc(indx)-gsubs(indx))./gsubt(indx))+1;
gsube=gsubs+(gsubc-1).*gsubt;

% search the nc file headers to locate the threads/processors
metricprefix='metric';
metriclist=dir([metric_dir,'/',metricprefix,'*.nc']);
n=1;
for i=1:length(metriclist)
    
    metricnm=[metric_dir,'/',metriclist(i).name];
    xyzs=double(nc_attget(metricnm,nc_global,'global_index_of_first_physical_points'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xyzc=double(nc_attget(metricnm,nc_global,'count_of_physical_points'));
    xc=xyzc(1);
    yc=xyzc(2);
    zc=xyzc(3);
    xarray=[xs:xs+xc-1];
    yarray=[ys:ys+yc-1];
    zarray=[zs:zs+zc-1];
   if (length(find(xarray>=gsubs(1)-1 & xarray<=gsube(1)-1)) ~= 0 && ...
       length(find(yarray>=gsubs(2)-1 & yarray<=gsube(2)-1)) ~= 0 && ...
       length(find(zarray>=gsubs(3)-1 & zarray<=gsube(3)-1)) ~= 0)
        px(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'px' )+2 : ...
                                         strfind(metriclist(i).name,'_py')-1));
        py(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'py' )+2 : ...
                                         strfind(metriclist(i).name,'_pz')-1));
        pz(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'pz' )+2 : ...
                                         strfind(metriclist(i).name,'.nc')-1));
        n=n+1;
    end
    
end

% retrieve the snapshot information
nthd=0;
for ip=1:length(px)
    
    nthd=nthd+1;
    
    metricnm=[metric_dir,'/',metricprefix,'_px',num2str(px(ip)),...
            '_py',num2str(py(ip)),'_pz',num2str(pz(ip)),'.nc'];
    xyzs=double(nc_attget(metricnm,nc_global,'global_index_of_first_physical_points'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xyzc=double(nc_attget(metricnm,nc_global,'count_of_physical_points'));
    xc=xyzc(1);
    yc=xyzc(2);
    zc=xyzc(3);
    xe=xs+xc-1;
    ye=ys+yc-1;
    ze=zs+zc-1;
    
    gxarray=gsubs(1):gsubt(1):gsube(1);
    gxarray=gxarray-1;
    gyarray=gsubs(2):gsubt(2):gsube(2);
    gyarray=gyarray-1;
    gzarray=gsubs(3):gsubt(3):gsube(3);
    gzarray=gzarray-1;
    
    metricinfo(nthd).thisid=[px(ip),py(ip),pz(ip)];
    i=find(gxarray>=xs & gxarray<=xe);
    j=find(gyarray>=ys & gyarray<=ye);
    k=find(gzarray>=zs & gzarray<=ze);
    metricinfo(nthd).indxs=[i(1),j(1),k(1)];
    metricinfo(nthd).indxe=[i(end),j(end),k(end)];
    metricinfo(nthd).indxc=metricinfo(nthd).indxe-metricinfo(nthd).indxs+1;
    
    metricinfo(nthd).subs=[ gxarray(i(1))-xs+1, ...
                            gyarray(j(1))-ys+1, ...
                            gzarray(k(1))-zs+1 ];
    metricinfo(nthd).subc=metricinfo(nthd).indxc;
    metricinfo(nthd).subt=gsubt;
    
    metricinfo(nthd).wsubs=double(nc_attget(metricnm,nc_global,'local_index_of_first_physical_points'))...
        +(metricinfo(nthd).subs-1).*metric_subt+1;
    metricinfo(nthd).wsubc=metricinfo(nthd).indxc;
    metricinfo(nthd).wsubt=metric_subt.*gsubt;
    
    metricinfo(nthd).tinv=snap_tinv;
    
    metricinfo(nthd).fnmprefix=metricprefix;
    
    metricinfo(nthd).ttriple=[gtstart,gtcount,gtstride];
    
end


end

