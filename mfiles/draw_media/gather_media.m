function [V,varargout] = gather_media(mediainfo,varnm,media_dir)

% check path exists
if ~ exist(media_dir,'dir')
   error([mfilename ': directory ' media_dir ' does not exist']);
end

% load
mediaprefix='media';
nthd=length(mediainfo);
for n=1:nthd
    
    n_i=mediainfo(n).thisid(1); n_j=mediainfo(n).thisid(2); n_k=mediainfo(n).thisid(3);
    i1=mediainfo(n).indxs(1); j1=mediainfo(n).indxs(2); k1=mediainfo(n).indxs(3);
    i2=mediainfo(n).indxe(1); j2=mediainfo(n).indxe(2); k2=mediainfo(n).indxe(3);
    subs=mediainfo(n).wsubs; subc=mediainfo(n).wsubc; subt=mediainfo(n).wsubt;
    fnm_media=[media_dir,'/',mediaprefix,'_px',num2str(n_i),'_py',num2str(n_j),'_pz',num2str(n_k),'.nc'];
    
    if ~ exist(fnm_media,'file')
       error([mfilename ': file ' fnm_media 'does not exist']);
    end

    V(k1:k2,j1:j2,i1:i2)=nc_varget(fnm_media,varnm, ...
                            fliplr(subs)-1,fliplr(subc),fliplr(subt));
                               
end

V=double(permute(V,[3,2,1]));

% pack varargout
if nargout>=2
    varargout(1)={varnm};
end


end
