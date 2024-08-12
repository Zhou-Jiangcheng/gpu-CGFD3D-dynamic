function [V] = gather_fault_final(output_dir,faultid,varnm,nproj,nprok)

  for kp=0:nprok-1      
    for jp=0:nproj-1
      % snapshot data
      faultstruct=dir([output_dir,'/','fault_i',num2str(faultid),'_px*_py',num2str(jp),'_pz',num2str(kp),'.nc']);
      faultnm=faultstruct.name;
      faultnm_dir=[output_dir,'/',faultnm];
      pnjstruct=nc_getdiminfo(faultnm_dir,'j');
      pnj=pnjstruct.Length;
      pnkstruct=nc_getdiminfo(faultnm_dir,'k');
      pnk=pnkstruct.Length;
      if jp==0
          VV=nc_varget(faultnm_dir,varnm,[0,0],[pnk,pnj],[1,1]);
      else
          VV0=nc_varget(faultnm_dir,varnm,[0,0],[pnk,pnj],[1,1]);
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

