function [n,s,d] = cal_basevectors(metric)

% n -> normal
% s -> strike
% d -> dip
[nj, nk] = size(metric.xix);

n = zeros(3, nj, nk);
s = zeros(3, nj, nk);
d = zeros(3, nj, nk);

n(1,:,:) = metric.xix;
n(2,:,:) = metric.xiy;
n(3,:,:) = metric.xiz;

s(1,:,:) = metric.x_et;
s(2,:,:) = metric.y_et;
s(3,:,:) = metric.z_et;

n_norm(:,:)=sqrt(n(1,:,:).^2+n(2,:,:).^2+n(3,:,:).^2);
s_norm(:,:)=sqrt(s(1,:,:).^2+s(2,:,:).^2+s(3,:,:).^2);

n(1,:,:) = squeeze(n(1,:,:)) ./ n_norm(:,:);
n(2,:,:) = squeeze(n(2,:,:)) ./ n_norm(:,:);
n(3,:,:) = squeeze(n(3,:,:)) ./ n_norm(:,:);

s(1,:,:) = squeeze(s(1,:,:)) ./ s_norm(:,:);
s(2,:,:) = squeeze(s(2,:,:)) ./ s_norm(:,:);
s(3,:,:) = squeeze(s(3,:,:)) ./ s_norm(:,:);
% d = n X s
d(1,:,:) = n(2,:,:).*s(3,:,:)-n(3,:,:).*s(2,:,:);
d(2,:,:) = n(3,:,:).*s(1,:,:)-n(1,:,:).*s(3,:,:);
d(3,:,:) = n(1,:,:).*s(2,:,:)-n(2,:,:).*s(1,:,:);

end
