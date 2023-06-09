function [V] = gather_fault_notime(output_dir,varnm,nproj,nprok)

  for kp=0:nprok-1      
    for jp=0:nproj-1
      % snapshot data
      nc_fnm_stuct=dir([output_dir,'/','fault_i*','_px*_py',num2str(jp),'_pz',num2str(kp),'.nc']);
      nc_fnm = [output_dir,'/', nc_fnm_stuct.name];

      finfo = ncinfo(nc_fnm);
      dimNames = {finfo.Dimensions.Name};

      jdimMatch = strncmpi(dimNames,'j',1);
      pnj       = finfo.Dimensions(jdimMatch).Length;

      kdimMatch = strncmpi(dimNames,'k',1);
      pnk       = finfo.Dimensions(kdimMatch).Length;

      if jp==0
          %-- attd: matlab ncread treats nc file dim order same to Fortran
          %VV=squeeze(ncread(nc_fnm,varnm,[1,1],[pnj,pnk],[1,1]));
          VV=squeeze(ncread(nc_fnm,varnm));
      else
          %VV0=squeeze(ncread(nc_fnm,varnm,[1,1],[pnj,pnk],[1,1]));
          VV0=squeeze(ncread(nc_fnm,varnm));
          VV=horzcat(VV,VV0);
      end
      
    end % end jp
    if kp==0
        V=VV;
    else
        V=vertcat(V,VV);
    end
  end  % end kp

end % end function
