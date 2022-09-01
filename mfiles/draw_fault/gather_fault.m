function [V, t] = gather_fault_x(output_dir,nlayer,varnm,nproj,nprok)



  for kp=0:nprok-1      
    for jp=0:nproj-1
      % snapshot data
      faultstruct=dir([output_dir,'/','fault_i*','_px*_py',num2str(jp),'_pz',num2str(kp),'.nc']);
      faultnm=faultstruct.name;
      faultnm_dir=[output_dir,'/',faultnm];
      pnjstruct=nc_getdiminfo(faultnm_dir,'j');
      pnj=pnjstruct.Length;
      pnkstruct=nc_getdiminfo(faultnm_dir,'k');
      pnk=pnkstruct.Length;
      if jp==0
          VV=squeeze(nc_varget(faultnm_dir,varnm,[nlayer-1,0,0],[1,pnk,pnj],[1,1,1]));
      else
          VV0=squeeze(nc_varget(faultnm_dir,varnm,[nlayer-1,0,0],[1,pnk,pnj],[1,1,1]));
          VV=horzcat(VV,VV0);
      end
      t=nc_varget(faultnm_dir,'time',[nlayer-1],[1]);
      
    end % end jp
    if kp==0
        V=VV;
    else
        V=vertcat(V,VV);
    end
  end  % end kp

end % end function
