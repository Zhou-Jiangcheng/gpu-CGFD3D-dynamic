function [nt] = fault_num_time(output_dir)

  % px0_py0_pz0.nc absolute exist
  faultstruct=dir([output_dir,'/','fault_i*','_px0_py0_pz0.nc']);
  faultnm=faultstruct.name;
  faultnm_dir=[output_dir,'/',faultnm];

  finfo = ncinfo(faultnm_dir);
  dimNames = {finfo.Dimensions.Name};
  dimMatch = strncmpi(dimNames,'time',1);

  nt = finfo.Dimensions(dimMatch).Length;

end % end function
