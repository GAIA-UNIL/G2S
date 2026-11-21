% A covariance given as a table (map): a damped hole effect, which the
% parametric structures cannot express. The table is a valid (positive
% definite) covariance: the product of an exponential covariance and the
% covariance of a random-phase cosine wave, sampled far enough for the
% exponential to have vanished at the edge of the table.
half = 80;                                    % lags -80..80; lag 0 at index half+1 (0-based: size/2)
[dx, dy] = meshgrid(-half:half, -half:half);  % dx: column shift, dy: row shift
direction = deg2rad(30);
table = exp(-hypot(dx, dy)/10) .* cos(2*pi*(dx*cos(direction) + dy*sin(direction))/25);

model = g2sCov.map(g2sCov.var('C', table));
disp(char(model))                             % 1*map(C)
disp(model.eval([0 0; 12 0; 25 0]))           % read from the table by covEval

args = model.args();
destination = nan(200, 200);
[sgsResult, ~] = g2s('-a', 'sgs', '-di', destination, '-dt', [0], args{:}, '-n', 40, '-s', 2, '-j', 0.5);
[fftmaResult, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-s', 2, '-j', 0.5);

figure;
subplot(1, 3, 1); imagesc(table); axis image; title('covariance map C');
subplot(1, 3, 2); imagesc(sgsResult.simulation); axis image; title('sgs with map(C)');
subplot(1, 3, 3); imagesc(fftmaResult.simulation); axis image; title('fftma with map(C)');
