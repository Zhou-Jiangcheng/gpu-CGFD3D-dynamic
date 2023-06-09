function [X,Y,Z] = gather_fault_coord(output_dir,nproj,nprok)

  for kp=0:nprok-1      
    for jp=0:nproj-1

      %-- fault slip file
      nc_fnm_stuct=dir([output_dir,'/','fault_i*','_px*_py',num2str(jp),'_pz',num2str(kp),'.nc']);
      nc_fnm = [output_dir,'/', nc_fnm_stuct.name];
      faultnm = nc_fnm_stuct.name;

      finfo = ncinfo(nc_fnm);
      dimNames = {finfo.Dimensions.Name};

      jdimMatch = strncmpi(dimNames,'j',1);
      pnj       = finfo.Dimensions(jdimMatch).Length;

      kdimMatch = strncmpi(dimNames,'k',1);
      pnk       = finfo.Dimensions(kdimMatch).Length;

      %-- i-index
      i_phys = str2num(faultnm( strfind(faultnm,'fault_i')+7 : strfind(faultnm,'_px')-1 ));

      % coordinate data
      ip=str2num(faultnm( strfind(faultnm,'px')+2 : strfind(faultnm,'_py')-1 ));
      coordnm=['coord','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      nc_fnm_coord=[output_dir,'/',coordnm];

      finfo = ncinfo(nc_fnm_coord);

      dimNames = {finfo.Dimensions.Name};
      kdimMatch = strncmpi(dimNames,'k',1);
      nk_coord = finfo.Dimensions(kdimMatch).Length;

      %idwithghost=double(nc_attget(faultnm_dir,nc_global,'i_index_with_ghosts_in_this_thread'));
      attNames = {finfo.Attributes.Name};
      dimMatch = strcmp(attNames,'local_index_of_first_physical_points');
      ghostwids = double(finfo.Attributes(dimMatch).Value);

      ibeg = ghostwids(1) + i_phys;
      jbeg = ghostwids(2) + 1;
      kbeg = ghostwids(3) + 1;

      if jp==0
          XX=squeeze(ncread(nc_fnm_coord,'x',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
          YY=squeeze(ncread(nc_fnm_coord,'y',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
          ZZ=squeeze(ncread(nc_fnm_coord,'z',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
      else
          XX0=squeeze(ncread(nc_fnm_coord,'x',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
          YY0=squeeze(ncread(nc_fnm_coord,'y',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
          ZZ0=squeeze(ncread(nc_fnm_coord,'z',[ibeg,jbeg,kbeg],[1,pnj,pnk],[1,1,1]));
          XX=horzcat(XX,XX0);
          YY=horzcat(YY,YY0);
          ZZ=horzcat(ZZ,ZZ0);
      end             
    end % end jp
    if kp==0
        X=XX;
        Y=YY;
        Z=ZZ;
    else
        X=vertcat(X,XX);
        Y=vertcat(Y,YY);
        Z=vertcat(Z,ZZ);
    end
  end  % end kp

end % end function


