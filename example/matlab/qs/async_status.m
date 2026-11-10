ti = single(imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff'));

submitted = g2s( ...
    '-a', 'qs', ...
    '-submitOnly', ...
    '-ti', ti, ...
    '-di', single(nan(200, 200)), ...
    '-dt', [0], ...
    '-k', 1.2, ...
    '-n', 50, ...
    '-j', 0.5);

job_id = submitted.job_id;
status = g2s('-statusOnly', job_id);
result = g2s('-waitAndDownload', job_id);

disp(['submitted ', num2str(job_id)]);
statusText = 'unknown';
if isfield(status, 'status')
    statusText = status.status;
end
progress = NaN;
if isfield(status, 'progress')
    progress = status.progress;
end
disp(['status ', statusText, ', progress ', num2str(progress)]);
disp(size(result.simulation));
