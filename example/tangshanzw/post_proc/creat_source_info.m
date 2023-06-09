clc;
clear;
close all;

set_mpath

%---------------------------------------------------------------
%- settings
%---------------------------------------------------------------

parfile    = '../run2/params.json'
output_dir = '../run2/output'

par = loadjson(parfile);
nproi=1;
nproj=par.number_of_mpiprocs_y;
nprok=par.number_of_mpiprocs_z;
j1 = par.fault_grid(1);
j2 = par.fault_grid(2);
k1 = par.fault_grid(3);
k2 = par.fault_grid(4);
dh = par.grid_generation_method.fault_plane.fault_inteval

nt = fault_num_time(output_dir)
skip_it = 3

%---------------------------------------------------------------
%- read rutpure results
%---------------------------------------------------------------

num_out_tdim = floor( nt / skip_it)
npt_j = j2 - j1 + 1
npt_k = k2 - k1 + 1

%-- pre-alloc
dyn_rake = zeros([npt_j, npt_k, num_out_tdim]);
dyn_rate = zeros([npt_j, npt_k, num_out_tdim]);
dyn_tdim = zeros([1,num_out_tdim]);

it_out = 0; 

for it = 1 : skip_it : nt

    it_out = it_out + 1;
  
    disp(['fd_out_it=',num2str(it), '  dyn_to_kin_it=',num2str(it_out)]);

    [Vs1,t] = gather_fault_1tlay(output_dir,it,'Vs1',nproj,nprok);
    [Vs2,t] = gather_fault_1tlay(output_dir,it,'Vs2',nproj,nprok);

    Vs1_fault = Vs1(j1:j2,k1:k2);
    Vs2_fault = Vs2(j1:j2,k1:k2);

    Vsabs = sqrt(Vs1_fault.^2+Vs2_fault.^2);
    rake  = atan2d( Vs2_fault(:,:),Vs1_fault(:,:) );

    dyn_rake(:,:,it_out) = rake;
    dyn_rate(:,:,it_out) = Vsabs;
    dyn_tdim(it_out    ) = t;

end

% get init t0
% (1*1 --> 1)
rupt_t = gather_fault_notime(output_dir,'init_t0',nproj,nprok);
dyn_t0 = rupt_t(j1:j2 , k1:k2);

%-- save to .mat file for later usage
save('dyn_vars.mat', ...
     'dyn_rake', ...
     'dyn_rate', ...
     'dyn_tdim', ...
     'dyn_t0');

