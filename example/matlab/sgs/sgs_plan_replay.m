% Plan once, replay later: the kriging weights of one path are written to a
% file (-wo), then replayed for more realisations (-wi) without solving any
% kriging system. The file is written and read by the server, so the path is
% a path on the server machine.
model = g2sCov.spherical('range', [50 25], 'angle', 45);
args = model.args();
destination = nan(300, 300);
plan = fullfile(tempdir, 'sgs_plan.bin');

[first, ~] = g2s('-a', 'sgs', '-di', destination, '-dt', [0], args{:}, ...
                 '-n', 30, '-nr', 8, '-s', 11, '-wo', plan, '-j', 0.5);

% Same inputs, same seed: realisations 8 to 39 of the same path
[extra, ~] = g2s('-a', 'sgs', '-di', destination, '-dt', [0], args{:}, ...
                '-n', 30, '-nr', 32, '-ri', 8, '-s', 11, '-wi', plan, '-j', 0.5);

% Replaying realisations 0 to 7 gives back the first run exactly
[again, ~] = g2s('-a', 'sgs', '-di', destination, '-dt', [0], args{:}, ...
                 '-n', 30, '-nr', 8, '-s', 11, '-wi', plan, '-j', 0.5);
fprintf('identical: %d\n', isequal(again.simulation, first.simulation));
