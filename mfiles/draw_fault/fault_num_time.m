function [nt] = fault_num_time(output_dir)

  % px0_py0_pz0.nc absolute exist
  faultstruct=dir([output_dir,'/','fault_i*','_px0_py0_pz0.nc']);
  faultnm=faultstruct.name;
  faultnm_dir=[output_dir,'/',faultnm];
  ntstruct=nc_getdiminfo(faultnm_dir,'time');
  nt=ntstruct.Length;

end % end function
