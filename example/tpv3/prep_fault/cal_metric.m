function metric = cal_metric(x, y, z);

[nj, nk] = size(x);
ny = nj + 6; % with ghost points number
nz = nk + 6;

x_g = zeros(ny, nz);
y_g = zeros(ny, nz);
z_g = zeros(ny, nz);

x_xi = zeros(nj, nk);
y_xi = zeros(nj, nk);
z_xi = zeros(nj, nk);
x_et = zeros(nj, nk);
y_et = zeros(nj, nk);
z_et = zeros(nj, nk);
x_zt = zeros(nj, nk);
y_zt = zeros(nj, nk);
z_zt = zeros(nj, nk);
jac  = zeros(nj, nk);
xix  = zeros(nj, nk);
xiy  = zeros(nj, nk);
xiz  = zeros(nj, nk);
etx  = zeros(nj, nk);
ety  = zeros(nj, nk);
etz  = zeros(nj, nk);
ztx  = zeros(nj, nk);
zty  = zeros(nj, nk);
ztz  = zeros(nj, nk);

for j = 1:nj
  for k = 1:nk
    x_g(j+3,k+3) = x(j,k);
    y_g(j+3,k+3) = y(j,k);
    z_g(j+3,k+3) = z(j,k);
  end
end
x_g = extend_symm(x_g);
y_g = extend_symm(y_g);
z_g = extend_symm(z_g);

% 6th order center finite difference
c1 = -0.02084;
c2 =  0.1667;
c3 = -0.7709;
c4 = 0;
c5 = 0.7709;
c6 = -0.1667;
c7 = 0.02084;
% c1 = -1/60;
% c2 =  3/20;
% c3 = -3/4;
% c4 = 0;
% c5 = 3/4;
% c6 = -3/20;
% c7 = 1/60;

x_xi(:,:) = 1.0;
y_xi(:,:) = 0.0;
z_xi(:,:) = 0.0;

for j = 4:ny-3;
  for k = 4:nz-3;

    x_et(j-3,k-3) = c1*x_g(j-3,k) + c2*x_g(j-2,k) + c3*x_g(j-1,k) +...
                    c5*x_g(j+1,k) + c6*x_g(j+2,k) + c7*x_g(j+3,k);
    y_et(j-3,k-3) = c1*y_g(j-3,k) + c2*y_g(j-2,k) + c3*y_g(j-1,k) +... 
                    c5*y_g(j+1,k) + c6*y_g(j+2,k) + c7*y_g(j+3,k);
    z_et(j-3,k-3) = c1*z_g(j-3,k) + c2*z_g(j-2,k) + c3*z_g(j-1,k) +...
                    c5*z_g(j+1,k) + c6*z_g(j+2,k) + c7*z_g(j+3,k);
    
    x_zt(j-3,k-3) = c1*x_g(j,k-3) + c2*x_g(j,k-2) + c3*x_g(j,k-1) +... 
                    c5*x_g(j,k+1) + c6*x_g(j,k+2) + c7*x_g(j,k+3);
    y_zt(j-3,k-3) = c1*y_g(j,k-3) + c2*y_g(j,k-2) + c3*y_g(j,k-1) +...
                    c5*y_g(j,k+1) + c6*y_g(j,k+2) + c7*y_g(j,k+3);
    z_zt(j-3,k-3) = c1*z_g(j,k-3) + c2*z_g(j,k-2) + c3*z_g(j,k-1) +... 
                    c5*z_g(j,k+1) + c6*z_g(j,k+2) + c7*z_g(j,k+3);
  end
end

jac = x_xi.*y_et.*z_zt+...
      x_zt.*y_xi.*z_et+...
      x_et.*y_zt.*z_xi-...
      x_xi.*y_zt.*z_et-...
      x_et.*y_xi.*z_zt-...
      x_zt.*y_et.*z_xi;


for j = 1:nj
  for k = 1:nk
    M = [ x_xi(j,k), x_et(j,k), x_zt(j,k);
          y_xi(j,k), y_et(j,k), y_zt(j,k);
          z_xi(j,k), z_et(j,k), z_zt(j,k);];
    N = inv(M);
    xix(j,k) = N(1,1);
    xiy(j,k) = N(1,2);
    xiz(j,k) = N(1,3);
    
    etx(j,k) = N(2,1);
    ety(j,k) = N(2,2);
    etz(j,k) = N(2,3);
    
    ztx(j,k) = N(3,1);
    zty(j,k) = N(3,2);
    ztz(j,k) = N(3,3);
end
end

metric.x_xi = x_xi;
metric.y_xi = y_xi;
metric.z_xi = z_xi;
metric.x_et = x_et;
metric.y_et = y_et;
metric.z_et = z_et;
metric.x_zt = x_zt;
metric.y_zt = y_zt;
metric.z_zt = z_zt;
metric.jac = jac;
metric.xix = xix;
metric.xiy = xiy;
metric.xiz = xiz;
metric.etx = etx;
metric.ety = ety;
metric.etz = etz;
metric.ztx = ztx;
metric.zty = zty;
metric.ztz = ztz;

end

function a = extend_symm(a)
[m, n] = size(a);
for i=1:3
  a(i,:) = 2*a(4,:) - a(2*4-i,:);
  a(:,i) = 2*a(:,4) - a(:,2*4-i);
  a(m-i+1,:) = 2*a(m-3,:) - a(2*(m-3)-(m-i+1),:); 
  a(:,n-i+1) = 2*a(:,n-3) - a(:,2*(n-3)-(n-i+1)); 
end

end
