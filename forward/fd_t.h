#ifndef FD_T_H
#define FD_T_H

/*******************************************************************************
 *macro for fd opterators
 *******************************************************************************/

// fault traction image coef
#define a_0   1.2500392377
#define a_1   1.5417647610
#define a_2  -0.3334118088
#define a_3   0.0416862855

//macdrp 2nd op

#define M_FD_SHIFT_PTR_MAC22(deriv, var_ptr, stride, FLAG) \
  ((FLAG==1) ? MAC22_F(deriv, var_ptr, stride) : MAC22_B(deriv, var_ptr, stride))

#define MAC22_F(deriv, var_ptr, stride)  \
  (deriv =  *(var_ptr + stride) - *(var_ptr) )

#define MAC22_B(deriv, var_ptr, stride)  \
  (deriv =  *(var_ptr) - *(var_ptr - stride) )

//macdrp 4th op
#define b_1  -7.0/6.0
#define b_2  8.0/6.0
#define b_3  -1.0/6.0

#define M_FD_SHIFT_PTR_MAC24(deriv, var_ptr, stride, FLAG) \
  ((FLAG==1) ? MAC24_F(deriv, var_ptr, stride) : MAC24_B(deriv, var_ptr, stride))

#define MAC24_F(deriv, var_ptr, stride)    \
  (deriv =  (b_1 * *(var_ptr))             \
           +(b_2 * *(var_ptr + stride))    \
           +(b_3 * *(var_ptr + 2*stride)))

#define MAC24_B(deriv, var_ptr, stride)    \
  (deriv = -(b_1 * *(var_ptr))             \
           -(b_2 * *(var_ptr - stride))    \
           -(b_3 * *(var_ptr - 2*stride)))

#define M_FD_VEC_24(deriv, var_ptr, FLAG) \
  ((FLAG==1) ? VEC_24_F(deriv, var_ptr) : VEC_24_B(deriv, var_ptr))

#define VEC_24_F(deriv, var_ptr)       \
  (deriv =  (b_1 * *(var_ptr))         \
           +(b_2 * *(var_ptr + 1))     \
           +(b_3 * *(var_ptr + 2)))

#define VEC_24_B(deriv, var_ptr)       \
  (deriv = -(b_3 * *(var_ptr))         \
           -(b_2 * *(var_ptr + 1))     \
           -(b_1 * *(var_ptr + 2)))

//macdrp normal op
#define c_1  -0.30874
#define c_2  -0.6326
#define c_3  1.233
#define c_4  -0.3334
#define c_5  0.04168

// use this MACDRP more faster. used by wavefield calculate.
#define M_FD_SHIFT_PTR_MACDRP_COEF(deriv, var_ptr, fd_shift, fd_coef)     \
  (deriv =  fd_coef[0] * *(var_ptr + fd_shift[0])                         \
          + fd_coef[1] * *(var_ptr + fd_shift[1])                         \
          + fd_coef[2] * *(var_ptr + fd_shift[2])                         \
          + fd_coef[3] * *(var_ptr + fd_shift[3])                         \
          + fd_coef[4] * *(var_ptr + fd_shift[4]))

// due to flag judge, more slower, but more convenient. used by fault calculate
#define M_FD_SHIFT_PTR_MACDRP(deriv, var_ptr, stride, FLAG) \
  ((FLAG==1) ? MACDRP_F(deriv, var_ptr, stride) : MACDRP_B(deriv, var_ptr, stride))

#define MACDRP_F(deriv, var_ptr, stride)    \
  (deriv = (c_1 * *(var_ptr - stride))      \
          +(c_2 * *(var_ptr))               \
          +(c_3 * *(var_ptr + stride))      \
          +(c_4 * *(var_ptr + 2*stride))    \
          +(c_5 * *(var_ptr + 3*stride)))

#define MACDRP_B(deriv, var_ptr, stride)    \
  (deriv = -(c_1 * *(var_ptr + stride))     \
           -(c_2 * *(var_ptr))              \
           -(c_3 * *(var_ptr - stride))     \
           -(c_4 * *(var_ptr - 2*stride))   \
           -(c_5 * *(var_ptr - 3*stride)))


#define M_FD_VEC_DRP(deriv, var_ptr, FLAG) \
  ((FLAG==1) ? VEC_DRP_F(deriv, var_ptr) : VEC_DRP_B(deriv, var_ptr))

#define VEC_DRP_F(deriv, var_ptr)     \
  (deriv =  (c_1 * *(var_ptr))        \
           +(c_2 * *(var_ptr + 1))    \
           +(c_3 * *(var_ptr + 2))    \
           +(c_4 * *(var_ptr + 3))    \
           +(c_5 * *(var_ptr + 4)))

#define VEC_DRP_B(deriv, var_ptr)     \
  (deriv = -(c_5 * *(var_ptr))        \
           -(c_4 * *(var_ptr + 1))    \
           -(c_3 * *(var_ptr + 2))    \
           -(c_2 * *(var_ptr + 3))    \
           -(c_1 * *(var_ptr + 4)))

#define M_FD_VEC(deriv, var_ptr, FLAG) \
  ((FLAG==1) ? VEC_F(deriv, var_ptr) : VEC_B(deriv, var_ptr))

#define VEC_F(deriv, var_ptr)         \
  (deriv =  (c_1 * *(var_ptr - 1))    \
           +(c_2 * *(var_ptr    ))    \
           +(c_3 * *(var_ptr + 1))    \
           +(c_4 * *(var_ptr + 2))    \
           +(c_5 * *(var_ptr + 3)))

#define VEC_B(deriv, var_ptr)          \
  (deriv = -(c_1 * *(var_ptr + 1))     \
           -(c_2 * *(var_ptr    ))     \
           -(c_3 * *(var_ptr - 1))     \
           -(c_4 * *(var_ptr - 2))     \
           -(c_5 * *(var_ptr - 3)))

//certer 6th op
//#define d_1  -0.02084
//#define d_2  0.1667
//#define d_3  -0.7709
//#define d_4  0.0
//#define d_5  0.7709
//#define d_6  -0.1667
//#define d_7  0.02084
#define d_1  -0.01666667
#define d_2   0.15000000
#define d_3  -0.75000000
#define d_4   0.0
#define d_5   0.75000000
#define d_6  -0.15000000
#define d_7   0.01666667

#define M_FD_SHIFT_PTR_CENTER(deriv, var_ptr, stride) \
 ( deriv =   (d_1 * *(var_ptr - 3*stride))            \
           + (d_2 * *(var_ptr - 2*stride))            \
           + (d_3 * *(var_ptr - stride))              \
           + (d_5 * *(var_ptr + stride))              \
           + (d_6 * *(var_ptr + 2*stride))            \
           + (d_7 * *(var_ptr + 3*stride)) )

/*******************************************************************************
 * structure for different fd schemes
 ******************************************************************************/

/*
 * elementary operator
 */

typedef struct
{
  int total_len;
  int left_len;
  int right_len;
  int *indx;
  int *shift;
  float *coef;
  int dir;
} fd_op_t; 

typedef struct {

  float CFL; // 1d cfl value for the scheme

  //----------------------------------------------------------------------------
  // Runge-Kutta time scheme
  //----------------------------------------------------------------------------

  int num_rk_stages;
  float *rk_a;
  float *rk_b;
  float *rk_rhs_time; // relative time for rhs eval

  // ghost point required 
  int fdx_nghosts;
  int fdy_nghosts;
  int fdz_nghosts;

  // max total len of op
  int fdx_max_len;
  int fdy_max_len;
  int fdz_max_len;

  int num_of_pairs;

  fd_op_t ***pair_fdx_op; // [pair][stage][1]
  fd_op_t ***pair_fdy_op;
  fd_op_t ***pair_fdz_op;

} fd_t;

typedef struct
{
  float *fdx_coef_d;
  float *fdy_coef_d;
  float *fdz_coef_d;

  int *fdx_indx_d;
  int *fdy_indx_d;
  int *fdz_indx_d;
  
  int *fdx_shift_d;
  int *fdy_shift_d;
  int *fdz_shift_d;
} fd_device_t;

/*******************************************************************************
 * function prototype
 ******************************************************************************/

int 
fd_set_macdrp(fd_t *fd);

#endif
