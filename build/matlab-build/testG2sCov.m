function testG2sCov(server, port)
%TESTG2SCOV Tests of g2sCov against test/geostat/cov_cases.txt (no server needed).
%   testG2sCov                 server-free tests; the covEval part runs when a server
%                              answers on localhost:8128 (skipped otherwise)
%   testG2sCov(host, port)     server for the covEval part
%   Errors at the end when a check failed.

if nargin < 1, server = 'localhost'; end
if nargin < 2, port = 8128; end
here = fileparts(mfilename('fullpath'));
addpath(here);
failures = {};
checks = 0;

	function check(condition, label)
		checks = checks + 1;
		if ~condition
			failures{end + 1} = label;
			fprintf(2, 'FAIL: %s\n', label);
		end
	end
	function checkText(actual, expected, label)
		check(ischar(actual) && strcmp(actual, expected), sprintf('%s: got "%s", expected "%s"', label, char(actual), expected));
	end
	function checkError(fun, fragment, label)
		try
			fun();
			check(false, sprintf('%s: no error, expected "%s"', label, fragment));
		catch err
			check(contains(err.message, fragment), sprintf('%s: error "%s" lacks "%s"', label, err.message, fragment));
		end
	end

% ---- every line of cov_cases.txt
casesFile = fullfile(here, '..', '..', 'test', 'geostat', 'cov_cases.txt');
lines = regexp(fileread(casesFile), '\r?\n', 'split');
count = 0;
for k = 1:numel(lines)
	line = lines{k};
	if isempty(strtrim(line)) || startsWith(strtrim(line), '#')
		continue
	end
	split = strfind(line, ' ==> ');
	source = strtrim(line(1:split(1) - 1));
	expected = strtrim(line(split(1) + 5:end));
	label = sprintf('cov_cases.txt line %d (%s)', k, source);
	count = count + 1;
	if startsWith(expected, '!error')
		fragment = strtrim(expected(numel('!error') + 1:end));
		checkError(@() g2sCov.parse(source), fragment, label);
	else
		try
			checkText(char(g2sCov.parse(source)), expected, label);
			checkText(char(g2sCov.parse(expected)), expected, [label ' (round trip)']);
		catch err
			check(false, sprintf('%s: unexpected error %s', label, err.message));
		end
	end
end
check(count > 100, 'cov_cases.txt has cases');

% ---- numbers
values = {0, '0'; -0, '0'; 1, '1'; -3, '-3'; 100000, '100000'; 0.1, '0.1'; 1e-5, '1e-05'; 1.5e16, '1.5e+16'; ...
	1/3, '0.3333333333333333'; 0.1 + 0.2, '0.30000000000000004'; 999999999999999, '999999999999999'; 1e15, '1e+15'};
for k = 1:size(values, 1)
	checkText(g2sCov.formatNumber(values{k, 1}), values{k, 2}, 'formatNumber');
end
rng(1);
for k = 1:500
	v = (rand - 0.5) * 10^(randi(40) - 20);
	check(str2double(g2sCov.formatNumber(v)) == v, sprintf('formatNumber round trip %.17g', v));
end

% ---- builders
m = 0.1*g2sCov.nugget() + 0.6*g2sCov.spherical('range', [40 20], 'angle', 30) + 0.3*g2sCov.exponential('range', 100);
checkText(char(m), '0.1*nugget + 0.6*spherical(range=[40, 20], angle=30) + 0.3*exponential(range=100)', 'example 1');
a = m.args();
check(isequal(a, {'-cov', char(m)}), 'example 1 args');

rangeMap = 25*ones(30, 40);
angleMap = zeros(30, 40);
nug = g2sCov.var('nug', 0.1);
L = g2sCov.var('L', rangeMap);
theta = g2sCov.var('theta', angleMap);
m = nug*g2sCov.nugget() + (1 - nug)*g2sCov.spherical('range', [L, L/2], 'angle', theta);
checkText(char(m), 'nug*nugget + (1 - nug)*spherical(range=[L, L/2], angle=theta)', 'example 2');
check(isequal(string(m), "nug*nugget + (1 - nug)*spherical(range=[L, L/2], angle=theta)"), 'string()');
a = m.args();
check(numel(a) == 11 && isequal(a(1:5), {'-cov', char(m), '-cvar', 'nug', '0.1'}) && isequal(a(6:7), {'-cvar', 'L'}) ...
	&& isequal(a{8}, rangeMap) && isequal(a(9:10), {'-cvar', 'theta'}) && isequal(a{11}, angleMap), 'example 2 args');

cases = {
	g2sCov.nugget(), '1*nugget'
	g2sCov.spherical('range', 10), '1*spherical(range=10)'
	g2sCov.exponential('scale', 5), '1*exponential(scale=5)'
	g2sCov.gaussian('Range', 12.5), '1*gaussian(range=12.5)'
	g2sCov.cubic('range', [10 20 30], 'angle', [10 20 30]), '1*cubic(range=[10, 20, 30], angle=[10, 20, 30])'
	g2sCov.matern('nu', 0.5, 'angle', 45, 'range', [30 10]), '1*matern(range=[30, 10], angle=45, nu=0.5)'
	g2sCov.matern('scale', [3 1], 'nu', 2.5), '1*matern(scale=[3, 1], nu=2.5)'
	g2sCov.stable('alpha', 2, 'scale', 4), '1*stable(scale=4, alpha=2)'
	0.7*g2sCov.map('C'), '0.7*map(C)'
	g2sCov.spherical('range', 10, 'angle', -30), '1*spherical(range=10, angle=-30)'
	2*g2sCov.nugget(), '2*nugget'
	g2sCov.nugget()*2, '2*nugget'
	g2sCov.nugget()/4, '0.25*nugget'
	0.5.*g2sCov.nugget(), '0.5*nugget'
	0.5*(g2sCov.nugget() + 2*g2sCov.spherical('range', 1)), '0.5*nugget + 1*spherical(range=1)'
	0 + g2sCov.nugget(), '1*nugget'
	single(0.5)*g2sCov.nugget(), '0.5*nugget'
	(1 - 0.1)*g2sCov.nugget(), '0.9*nugget'
	g2sCov.exponential('range', 1/3), '1*exponential(range=0.3333333333333333)'
	g2sCov.spherical('range', {40, 20}), '1*spherical(range=[40, 20])'
	};
for k = 1:size(cases, 1)
	checkText(char(cases{k, 1}), cases{k, 2}, sprintf('builder %d', k));
	checkText(char(g2sCov.parse(cases{k, 2})), cases{k, 2}, sprintf('builder %d round trip', k));
end

a = g2sCov.var('a', 1); b = g2sCov.var('b', 1); c = g2sCov.var('c', 1); d = g2sCov.var('d', 1);
L = g2sCov.var('L', 1); R = g2sCov.var('R', 3); nuMap = g2sCov.var('nu_map', 1.5);
cases = {
	a*b*g2sCov.spherical('range', L), 'a*b*spherical(range=L)'
	a/2*g2sCov.nugget(), 'a/2*nugget'
	(a + b)/(c*d)*g2sCov.nugget(), '(a + b)/(c*d)*nugget'
	g2sCov.spherical('range', a - b), '1*spherical(range=a - b)'
	g2sCov.spherical('range', a - (b - c)), '1*spherical(range=a - (b - c))'
	g2sCov.spherical('range', (a - b) - c), '1*spherical(range=a - b - c)'
	g2sCov.spherical('range', a*(b*c)), '1*spherical(range=a*(b*c))'
	g2sCov.spherical('range', a./(b./c)), '1*spherical(range=a/(b/c))'
	g2sCov.spherical('range', min(L, 50), 'angle', -a), '1*spherical(range=min(L, 50), angle=-a)'
	g2sCov.exponential('range', sqrt(L)*2 + max(a, b)), '1*exponential(range=sqrt(L)*2 + max(a, b))'
	g2sCov.exponential('range', exp(log(L))), '1*exponential(range=exp(log(L)))'
	g2sCov.gaussian('range', 2*-L), '1*gaussian(range=2*-L)'
	g2sCov.gaussian('range', -(L - 2*L)), '1*gaussian(range=-(L - 2*L))'
	g2sCov.exponential('range', abs(L)), '1*exponential(range=abs(L))'
	(1 - nug)*g2sCov.matern('range', R, 'nu', nuMap), '(1 - nug)*matern(range=R, nu=nu_map)'
	g2sCov.exponential('range', L + (-2)), '1*exponential(range=L + -2)'
	g2sCov.exponential('range', 2 - (-L)), '1*exponential(range=2 - -L)'
	g2sCov.spherical('range', [L, 20]), '1*spherical(range=[L, 20])'
	a*g2sCov.nugget()*b, 'b*a*nugget'
	};
for k = 1:size(cases, 1)
	checkText(char(cases{k, 1}), cases{k, 2}, sprintf('expression %d', k));
	checkText(char(g2sCov.parse(cases{k, 2})), cases{k, 2}, sprintf('expression %d round trip', k));
end

% automatic names, shared variables, regions, parse with bindings
sill = 0.5*ones(4, 5); rng10 = 10*ones(4, 5);
m = sill*g2sCov.nugget() + g2sCov.exponential('range', rng10);
checkText(char(m), '_a1*nugget + 1*exponential(range=_a2)', 'automatic names');
a = m.args();
check(isequal(a([3 4 6 7]), {'-cvar', '_a1', '-cvar', '_a2'}) && isequal(a{5}, sill) && isequal(a{8}, rng10), 'automatic args');
m = g2sCov.var('_a1', 2)*g2sCov.nugget() + g2sCov.exponential('range', rng10);
checkText(char(m), '_a1*nugget + 1*exponential(range=_a2)', 'automatic names avoid user names');

L = g2sCov.var('L', 30);
m0 = 0.2*g2sCov.nugget() + 0.8*g2sCov.spherical('range', L);
m1 = g2sCov.exponential('range', L/2, 'angle', g2sCov.var('theta', -15));
index = zeros(3, 4); index(2:end, :) = 1;
s = g2sCov.regions({m0, m1}, index);
a = s.args();
check(isequal(a(1:3), {'-cov', '0.2*nugget + 0.8*spherical(range=L)', '1*exponential(range=L/2, angle=theta)'}), 'regions -cov');
check(isequal(a(4:8), {'-cvar', 'L', '30', '-cvar', 'theta=-15'}), 'regions -cvar');
check(strcmp(a{9}, '-ii') && isequal(double(a{10}), index), 'regions -ii');
checkError(@() g2sCov.regions({m0, m1}, index + 1), 'only 2 model(s)', 'regions index range');

rm = 12*ones(5, 6);
m = g2sCov.parse('nug*nugget + (1-nug)*exponential(range=L)', 'nug', 0.25, 'L', rm);
checkText(char(m), 'nug*nugget + (1 - nug)*exponential(range=L)', 'parse with bindings');
a = m.args();
check(isequal(a(3:5), {'-cvar', 'nug', '0.25'}) && isequal(a{8}, rm), 'parse bindings args');
[names, ~] = m.variables();
check(isequal(names, {'nug', 'L'}), 'variables()');
checkError(@() args(g2sCov.parse('nug*nugget')), 'has no value', 'unbound variable');
checkError(@() g2sCov.parse('nug*nugget', 'other', 1), 'does not use it', 'unused binding');
joined = g2sCov.parse('exponential(range=L)') + g2sCov.gaussian('range', g2sCov.var('L', 4));
a = joined.args();
check(isequal(a(end - 2:end), {'-cvar', 'L', '4'}), 'unbound takes the bound value');

% ---- validation
checkError(@() g2sCov.var('range', 1), 'reserved word', 'reserved range');
checkError(@() g2sCov.var('Spherical', 1), 'reserved word', 'reserved Spherical');
checkError(@() g2sCov.var('H', 1), 'reserved word', 'reserved H');
checkError(@() g2sCov.var('2L', 1), 'not a valid variable name', 'invalid name');
checkError(@() g2sCov.spherical(), 'needs range= or scale=', 'no range');
checkError(@() g2sCov.spherical('range', 1, 'scale', 2), 'given twice', 'range and scale');
checkError(@() g2sCov.spherical('range', -10), 'range/scale must be > 0', 'negative range');
checkError(@() g2sCov.spherical('range', [10 0]), 'range/scale must be > 0', 'zero range');
checkError(@() g2sCov.spherical('range', {1, 2, 3, 4, 5}), 'more than 4 values', 'too many ranges');
checkError(@() g2sCov.spherical('range', 1, 'angle', [1 2]), 'angle takes one value (2D) or three values (3D)', 'two angles');
checkError(@() g2sCov.spherical('range', 1, 'foo', 2), 'unknown keyword ''foo''', 'unknown keyword');
checkError(@() g2sCov.matern('range', 10), 'matern needs nu=', 'no nu');
checkError(@() g2sCov.matern('range', 10, 'nu', 0), 'nu must be in (0, 100]', 'nu 0');
checkError(@() g2sCov.matern('range', 10, 'nu', 101), 'nu must be in (0, 100]', 'nu 101');
checkError(@() g2sCov.matern('range', 10, 'nu', [1 2]), 'takes a single value', 'nu vector');
checkError(@() g2sCov.stable('range', 10, 'alpha', 2.5), 'alpha must be in (0, 2]', 'alpha 2.5');
checkError(@() g2sCov.spherical('range', 1i), 'must be a number', 'complex input rejected');
checkError(@() -0.5*g2sCov.nugget(), 'partial sill must be >= 0', 'negative sill');
checkError(@() g2sCov.nugget() - g2sCov.spherical('range', 1), 'cannot be subtracted', 'subtraction');
checkError(@() g2sCov.nugget()*g2sCov.spherical('range', 1), 'cannot be multiplied', 'model product');
checkError(@() g2sCov.nugget() + 1, 'cannot be added', 'model plus number');
checkError(@() [g2sCov.nugget(), g2sCov.nugget()], 'cannot be concatenated', 'model concatenation');
checkError(@() g2sCov.spherical('range', 'L'), 'text', 'text parameter');
checkError(@() args(g2sCov.exponential('range', g2sCov.var('L', 10)) + g2sCov.gaussian('range', g2sCov.var('L', 20))), ...
	'defined twice with different values', 'two values');
checkError(@() char(g2sCov.var('C', 1)*g2sCov.map(g2sCov.var('C', ones(3)))), 'map table and as a parameter', 'map conflict');
checkError(@() args(g2sCov.matern('range', 10, 'nu', g2sCov.var('nu_v', 200))), 'nu must be in (0, 100]', 'bound nu');
checkError(@() args((g2sCov.var('s', 0.5) - 1)*g2sCov.nugget()), 'partial sill must be >= 0', 'bound sill');
checkError(@() args(g2sCov.exponential('range', g2sCov.var('L', 4) - 5)), 'range/scale must be > 0', 'bound range');
args(g2sCov.exponential('range', g2sCov.var('L', -ones(2, 2)))); % maps are left to the server
check(true, 'maps are not checked');

% ---- precision: constants are passed as round-trip text
values = [0.1, 1.2345678901234567e-7, 123456.78901234567, 2/3, 1e-12, 6.02214076e23];
for k = 1:numel(values)
	a = args(g2sCov.var('s', values(k))*g2sCov.nugget());
	check(ischar(a{5}) && str2double(a{5}) == values(k), sprintf('precision of %.17g', values(k)));
	text = char(values(k)*g2sCov.nugget());
	check(str2double(text(1:strfind(text, '*') - 1)) == values(k), sprintf('precision in text %.17g', values(k)));
end
a = args(g2sCov.spherical('range', 10, 'angle', g2sCov.var('theta', -30.5)));
check(isequal(a, {'-cov', '1*spherical(range=10, angle=theta)', '-cvar', 'theta=-30.5'}), 'negative constant');

% ---- covEval through a server (skipped without one)
serverChecked = false;
if any(exist('g2s', 'file') == [2 3])
	options = {'-sa', server, '-p', port, '-TO', 5000};
	try
		probe = g2sCov.nugget();
		probe.eval(0, options{:});
		serverChecked = true;
	catch err
		fprintf('server part skipped: %s\n', err.message);
	end
else
	fprintf('server part skipped: the g2s MEX file is not on the path\n');
end
if serverChecked
	m = 0.1*g2sCov.nugget() + 0.9*g2sCov.exponential('range', 30);
	h = [0; 10; 30; 100];
	expected = [1; 0.9*exp(-3*h(2:end)/30)];
	cv = m.eval(h, options{:});
	gv = m.variogram(h, options{:});
	check(max(abs(cv - expected)) < 1e-6, 'eval exponential');
	check(max(abs(gv - (1 - expected))) < 1e-6, 'variogram exponential');
	sph = @(t) (t < 1).*(1 - 1.5*t + 0.5*t.^3);
	sm = g2sCov.spherical('range', [40 20]);
	cv = sm.eval([20 0; 0 20; 0 10], options{:});
	check(max(abs(cv - [sph(0.5); 0; sph(0.5)])) < 1e-6, 'axis convention: range(1) along columns');
	cv = sm.eval([20 0], 'ndim', 2, options{:});
	check(abs(cv - sph(0.5)) < 1e-6, 'single 2D lag with ndim');
	grid = reshape(5:16, 4, 3).'; % 3 x 4, grid(2, 3) = 11 at 0-based (column 2, row 1)
	vm = g2sCov.exponential('range', g2sCov.var('L', grid));
	cv = vm.eval([0 0; 10 0], 'pos', [2 1], options{:});
	check(max(abs(cv - [1; exp(-3*10/11)])) < 1e-6, 'spatially varying with pos');
	s = 1.2345678e-7;
	pm = g2sCov.var('s', s)*g2sCov.nugget();
	cv = pm.eval(0, options{:});
	check(abs(cv/s - 1) < 1e-6, 'precision through the interface');
end

fprintf('testG2sCov: %d checks, %d failures%s\n', checks, numel(failures), ...
	repmat(' (server part skipped)', 1, ~serverChecked));
if ~isempty(failures)
	error('testG2sCov:failed', '%d check(s) failed', numel(failures));
end
end
