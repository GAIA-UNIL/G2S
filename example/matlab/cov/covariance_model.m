% Build a covariance model with g2sCov, look at its text and its g2s
% arguments, and evaluate it on the server (covEval) along two directions.

% 10% nugget and a spherical structure: practical range 40 cells along the
% major axis, 15 along the minor axis, major axis turned by 30 degrees
model = 0.1*g2sCov.nugget() + 0.9*g2sCov.spherical('range', [40 15], 'angle', 30);
disp(char(model))   % 0.1*nugget + 0.9*spherical(range=[40, 15], angle=30)
args = model.args() % {'-cov', '0.1*nugget + 0.9*spherical(range=[40, 15], angle=30)'}

% Lags are [column shift, row shift] (component 0 = columns)
distance = (0:50)';
theta = deg2rad(30);
major = [distance*cos(theta), distance*sin(theta)];
minor = [-distance*sin(theta), distance*cos(theta)];
gammaMajor = model.variogram(major);   % runs covEval on the G2S server
gammaMinor = model.variogram(minor);

% The same evaluation without the helper
[direct, ~] = g2s('-a', 'covEval', '-cov', char(model), '-lag', major, '-vario');
assert(max(abs(double(direct.variogram(:)) - gammaMajor)) < 1e-6);

figure;
plot(distance, gammaMajor, distance, gammaMinor);
xlabel('lag (cells)');
ylabel('variogram');
legend('major axis (30 degrees)', 'minor axis (120 degrees)');
title(char(model));
