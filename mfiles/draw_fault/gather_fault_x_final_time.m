function [V, X, Y, Z] = gather_fault_x_final_time(output_dir,varnm,nproj,nprok)



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
          VV=nc_varget(faultnm_dir,varnm,[0,0],[pnk,pnj],[1,1]);
      else
          VV0=nc_varget(faultnm_dir,varnm,[0,0],[pnk,pnj],[1,1]);
          VV=horzcat(VV,VV0);
      end
      
      % coordinate data
      ip=str2num(faultnm( strfind(faultnm,'px')+2 : strfind(faultnm,'_py')-1 ));
      coordnm=['coord','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      coordnm_dir=[output_dir,'/',coordnm];
      idwithghost=double(nc_attget(faultnm_dir,nc_global,'i_index_with_ghosts_in_this_thread'));
      coorddimstruct=nc_getdiminfo(coordnm_dir,'k');
      faultdimstruct=nc_getdiminfo(faultnm_dir,'k');
      ghostp=(coorddimstruct.Length-faultdimstruct.Length)/2;
      if jp==0
          XX=squeeze(nc_varget(coordnm_dir,'x',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
          YY=squeeze(nc_varget(coordnm_dir,'y',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
          ZZ=squeeze(nc_varget(coordnm_dir,'z',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
      else
          XX0=squeeze(nc_varget(coordnm_dir,'x',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
          YY0=squeeze(nc_varget(coordnm_dir,'y',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
          ZZ0=squeeze(nc_varget(coordnm_dir,'z',[ghostp,ghostp,idwithghost],[pnk,pnj,1],[1,1,1]));
          XX=horzcat(XX,XX0);
          YY=horzcat(YY,YY0);
          ZZ=horzcat(ZZ,ZZ0);
      end             
    end % end jp
    if kp==0
        V=VV;
        X=XX;
        Y=YY;
        Z=ZZ;
    else
        V=vertcat(V,VV);
        X=vertcat(X,XX);
        Y=vertcat(Y,YY);
        Z=vertcat(Z,ZZ);
    end
  end  % end kp

end % end function

