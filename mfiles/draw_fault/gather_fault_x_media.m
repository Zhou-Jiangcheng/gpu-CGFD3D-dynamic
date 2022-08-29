function [V] = gather_fault_x_media(output_dir,varnm,nproj,nprok)



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
      % coordinate data
      ip=str2num(faultnm( strfind(faultnm,'px')+2 : strfind(faultnm,'_py')-1 ));
      medianm=['media','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      medianm_dir=[output_dir,'/',medianm];
      idwithghost=double(nc_attget(faultnm_dir,nc_global,'i_index_with_ghosts_in_this_thread'));
      mediadimstruct=nc_getdiminfo(medianm_dir,'k');
      faultdimstruct=nc_getdiminfo(faultnm_dir,'k');
      ghostp=(mediadimstruct.Length-faultdimstruct.Length)/2;
      if jp==0
          VV=squeeze(nc_varget(medianm_dir,varnm,[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
      else
          VV0=squeeze(nc_varget(medianm_dir,varnm,[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
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

