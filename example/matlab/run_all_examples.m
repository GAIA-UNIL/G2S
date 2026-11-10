function run_all_examples()
% Run all MATLAB examples in isolated workspaces.

rootDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(rootDir));
legacyDir = fullfile(repoRoot, 'legacy_example', 'matlab');
selfPath = [mfilename('fullpath') '.m'];
expectedFailures = {
    fullfile(rootDir, 'reporting', 'reporting_probe.m'), ...
    fullfile(legacyDir, 'reporting_probe.m')
};

scripts = collectExampleScripts(rootDir, selfPath);
if exist(legacyDir, 'dir') == 7
    scripts = [scripts collectExampleScripts(legacyDir, selfPath)];
    scripts = sort(scripts);
end

if isempty(scripts)
    error('No MATLAB examples found under %s', rootDir);
end

fprintf('Running %d MATLAB example(s) from %s\n', numel(scripts), repoRoot);

passCount = 0;
expectedFailureCount = 0;
failureCount = 0;
unexpectedPassCount = 0;

for index = 1:numel(scripts)
    scriptPath = scripts{index};
    relativePath = erase(scriptPath, [repoRoot filesep]);
    expectedFailure = isExpectedFailure(scriptPath, expectedFailures);

    fprintf('[RUN] %s\n', relativePath);
    tic;
    [completed, exception, reachedExpectedMarker] = executeExample(scriptPath, expectedFailure);
    elapsed = toc;

    if completed
        if expectedFailure
            unexpectedPassCount = unexpectedPassCount + 1;
            fprintf('[UNEXPECTED PASS] %s in %.1fs\n', relativePath, elapsed);
        else
            passCount = passCount + 1;
            fprintf('[PASS] %s in %.1fs\n', relativePath, elapsed);
        end
    elseif expectedFailure && reachedExpectedMarker
        expectedFailureCount = expectedFailureCount + 1;
        fprintf('[EXPECTED FAIL] %s in %.1fs\n', relativePath, elapsed);
        fprintf('%s\n', exception.message);
    else
        failureCount = failureCount + 1;
        fprintf('[FAIL] %s in %.1fs\n', relativePath, elapsed);
        fprintf('%s\n', getReport(exception, 'extended', 'hyperlinks', 'off'));
    end
end

fprintf('\nSummary: %d passed, %d expected failure(s), %d failed, %d unexpected pass(es).\n', ...
    passCount, expectedFailureCount, failureCount, unexpectedPassCount);

if failureCount > 0 || unexpectedPassCount > 0
    error('One or more MATLAB examples did not complete as expected.');
end

end

function [completed, exception, reachedExpectedMarker] = executeExample(scriptPath, expectedFailure)
    completed = false;
    exception = [];
    reachedExpectedMarker = false;
    originalDir = pwd;
    cleanup = onCleanup(@() cd(originalDir));

    try
        cd(fileparts(scriptPath));
        runScriptInIsolatedWorkspace(scriptPath);
        completed = true;
    catch caughtException
        exception = caughtException;
    end

    if expectedFailure
        reachedExpectedMarker = ~isempty(exception) && contains(exception.message, 'report probe emitted a fatal error');
    end
end

function runScriptInIsolatedWorkspace(scriptPath)
    [~, scriptStem, ~] = fileparts(scriptPath);
    if isvarname(scriptStem)
        run(scriptPath);
    else
        scriptText = fileread(scriptPath);
        eval(scriptText);
    end
end

function scripts = collectExampleScripts(folder, selfPath)
    scripts = {};
    entries = dir(folder);
    for index = 1:numel(entries)
        entry = entries(index);
        if strcmp(entry.name, '.') || strcmp(entry.name, '..')
            continue;
        end

        entryPath = fullfile(folder, entry.name);
        if entry.isdir
            scripts = [scripts collectExampleScripts(entryPath, selfPath)]; %#ok<AGROW>
        elseif endsWith(entry.name, '.m') && ~strcmp(entryPath, selfPath)
            scripts{end + 1} = entryPath; %#ok<AGROW>
        end
    end
    scripts = sort(scripts);
end

function result = isExpectedFailure(scriptPath, expectedFailures)
    result = any(strcmp(scriptPath, expectedFailures));
end
