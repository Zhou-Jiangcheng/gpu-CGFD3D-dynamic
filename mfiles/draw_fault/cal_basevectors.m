function [n,m,l] = cal_basevectors(metric)

[nj, nk] = size(metric.xix);

n = zeros(3, nj, nk);
m = zeros(3, nj, nk);
l = zeros(3, nj, nk);

n(1,:,:) = metric.xix;
n(2,:,:) = metric.xiy;
n(3,:,:) = metric.xiz;

m(1,:,:) = metric.x_et;
m(2,:,:) = metric.y_et;
m(3,:,:) = metric.z_et;

n_norm(:,:)=sqrt(n(1,:,:).^2+n(2,:,:).^2+n(3,:,:).^2);
m_norm(:,:)=sqrt(m(1,:,:).^2+m(2,:,:).^2+m(3,:,:).^2);

n(1,:,:) = squeeze(n(1,:,:)) ./ n_norm(:,:);
n(2,:,:) = squeeze(n(2,:,:)) ./ n_norm(:,:);
n(3,:,:) = squeeze(n(3,:,:)) ./ n_norm(:,:);

m(1,:,:) = squeeze(m(1,:,:)) ./ m_norm(:,:);
m(2,:,:) = squeeze(m(2,:,:)) ./ m_norm(:,:);
m(3,:,:) = squeeze(m(3,:,:)) ./ m_norm(:,:);
% l = n X m
l(1,:,:) = n(2,:,:).*m(3,:,:)-n(3,:,:).*m(2,:,:);
l(2,:,:) = n(3,:,:).*m(1,:,:)-n(1,:,:).*m(3,:,:);
l(3,:,:) = n(1,:,:).*m(2,:,:)-n(2,:,:).*m(1,:,:);

end
