%-- strike line, given along strike direction
%fault_strike_file = 'tangshan_strike.txt'
fault_strike_file = 'tangshan_strike_reverse.txt'
%-- dip line, give along dip direction
fault_dip_file    = 'tangshan_dip.txt'
%-- estimated strike angle for local coordinate
strike_estimate = 220
%-- apply smooth or not
is_smooth_stirke = 1
%-- x-axis is defined positive to hanging wall direction
DX   = 90.0
%-- discrete size along strike
DLEN = 90.0
%-- discrete size along dip
DWID = 90.0
%-- DX,DLEN and DWID should be eqaul, which is assumed in some parts
DEPTH_TO_TOP   = 0e3
HYPO_ALONG_STK_POINTS = 200 % without padding points
HYPO_DOWN_DIP_POINTS  = 100 % without padding points
%-- number of grid points padding at begin of strike
NPAD_LEN_START = 100
%-- number of grid points padding at end of strike
NPAD_LEN_END   = 100
%-- number of grid points padding at end of dip
NPAD_DIP       = 100

%-- mu_s
FRICTION_STATIC  = 0.357
%-- mu_d
FRICTION_DYNAMIC = 0.275 
Dc = 0.4
%-- vertical stress
SV = 120.0
SV2SHMAX = 1.88
SV2SHMIN = 0.38
SHMAX_AZIMUTH = 87.0
nucleation_size = 1500.0

fnm_fault_grid   = './fault_coord.nc'
fnm_fault_stress = './init_stress.nc'

%-- info file for other program
fnm_log_grid   = './log_grid.txt'
fnm_log_stress = './log_stress.txt'

