#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from typing import Any, List, Sequence, Tuple


DATA_DIR = "/tmp/G2S/data"
DEFAULT_HOST = "localhost"
DEFAULT_PORT_START = 8130


def parse_arg_entries(argv: Sequence[str]) -> List[Tuple[str, List[str]]]:
	entries: List[Tuple[str, List[str]]] = []
	i = 1
	while i < len(argv):
		token = argv[i]
		if not token.startswith("-"):
			i += 1
			continue
		key = token
		i += 1
		values: List[str] = []
		while i < len(argv) and not argv[i].startswith("-"):
			values.append(argv[i])
			i += 1
		entries.append((key, values))
	return entries


def get_first_value(entries: Sequence[Tuple[str, List[str]]], keys: Sequence[str]) -> str:
	for key, values in entries:
		if key in keys and values:
			return values[0]
	return ""


def get_all_values(entries: Sequence[Tuple[str, List[str]]], key: str) -> List[str]:
	out: List[str] = []
	for current_key, values in entries:
		if current_key == key:
			out.extend(values)
	return out


def normalize_job_id(value: Any) -> str:
	if isinstance(value, str):
		value = value.strip()
		if value == "":
			raise ValueError("empty job id in -job_grid_json")
		parsed = int(value)
	elif isinstance(value, int):
		parsed = value
	elif isinstance(value, float):
		if not value.is_integer():
			raise ValueError("non integer job id in -job_grid_json")
		parsed = int(value)
	else:
		raise ValueError("unsupported job id type in -job_grid_json")

	if parsed < 0:
		raise ValueError("negative job id in -job_grid_json")
	return str(parsed)


def flatten_row_major(node: Any, out: List[str]) -> None:
	if isinstance(node, list):
		for child in node:
			flatten_row_major(child, out)
		return
	out.append(normalize_job_id(node))


def build_like(template: Any, values: List[str], index_ref: List[int]) -> Any:
	if isinstance(template, list):
		return [build_like(child, values, index_ref) for child in template]
	value = values[index_ref[0]]
	index_ref[0] += 1
	return value


def data_name_exists(name: str) -> bool:
	return os.path.exists(os.path.join(DATA_DIR, f"{name}.bgrid.gz")) or os.path.exists(
		os.path.join(DATA_DIR, f"{name}.bgrid")
	)


def get_default_ti(job_id: str) -> List[str]:
	ti_values: List[str] = []
	idx = 0
	while True:
		candidate = f"input_ti_{idx}_{job_id}"
		if not data_name_exists(candidate):
			break
		ti_values.append(candidate)
		idx += 1
	return ti_values


def get_default_di(job_id: str) -> str:
	candidate = f"input_di_{job_id}"
	if data_name_exists(candidate):
		return candidate
	return ""


def should_skip_forwarded_arg(key: str) -> bool:
    # Keep this list minimal: only args consumed by this wrapper
    # or arguments that must be controlled per child command.
    consumed = {
        "-a",
        "-job_grid_json",
        "-job_grid",
        "-jg",
        "-eg",
        "-endpoint_grid_json",
        "-ti",
        "-di",
        "-r",
    }
    return key in consumed


def to_cmd_args(entries: Sequence[Tuple[str, List[str]]]) -> List[str]:
	cmd_args: List[str] = []
	for key, values in entries:
		if should_skip_forwarded_arg(key):
			continue
		cmd_args.append(key)
		cmd_args.extend(values)
	return cmd_args


def ensure_job_inputs(job_id: str, local_ti: List[str], local_di: List[str]) -> Tuple[List[str], List[str]]:
	ti = local_ti if local_ti else get_default_ti(job_id)
	default_di = get_default_di(job_id)
	di = local_di if local_di else ([default_di] if default_di else [])
	if not ti:
		raise RuntimeError(f"no TI found for job {job_id} (missing -ti and missing input_ti_*_{job_id} links)")
	if not di:
		raise RuntimeError(f"no DI found for job {job_id} (missing -di and missing input_di_{job_id} link)")
	return ti, di


def main() -> int:
    start_time = time.time()
    entries = parse_arg_entries(sys.argv)
    report_path = get_first_value(entries, ("-r",))
    report_file = None
    report_should_close = False
    if report_path and report_path != "stderr" and report_path != "stdout":
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report_file = open(report_path, "a", buffering=1)
        report_should_close = True
    elif report_path == "stdout":
        report_file = sys.stdout
    else:
        report_file = sys.stderr

    def report(message: str) -> None:
        report_file.write(message + "\n")

    report(" ".join(sys.argv))

    job_grid_json = get_first_value(entries, ("-job_grid_json", "-job_grid", "-jg"))
    if not job_grid_json:
        report("qs_decentralized: missing -job_grid_json")
        if report_should_close:
            report_file.close()
        return 2

    try:
        job_grid = json.loads(job_grid_json)
    except Exception as exc:
        report(f"qs_decentralized: invalid -job_grid_json ({exc})")
        if report_should_close:
            report_file.close()
        return 2

    job_ids: List[str] = []
    try:
        flatten_row_major(job_grid, job_ids)
    except Exception as exc:
        report(f"qs_decentralized: invalid job id payload ({exc})")
        if report_should_close:
            report_file.close()
        return 2
    if not job_ids:
        report("qs_decentralized: empty -job_grid_json")
        if report_should_close:
            report_file.close()
        return 2

    endpoints = [f"{DEFAULT_HOST}:{DEFAULT_PORT_START + idx}" for idx in range(len(job_ids))]
    endpoint_grid = build_like(job_grid, endpoints, [0])
    endpoint_grid_json = json.dumps(endpoint_grid, separators=(",", ":"))

    local_ti = get_all_values(entries, "-ti")
    local_di = get_all_values(entries, "-di")
    forwarded = to_cmd_args(entries)

    all_cmds: List[Tuple[str, List[str]]] = []
    for job_id in job_ids:
        ti_values, di_values = ensure_job_inputs(job_id, local_ti, local_di)
        cmd: List[str] = ["./qs", "-r", f"/tmp/G2S/logs/{job_id}.log", "-ti"]
        cmd.extend(ti_values)
        cmd.append("-di")
        cmd.extend(di_values)
        cmd.extend(forwarded)
        cmd.extend(["-job_grid_json", job_grid_json, "-eg", endpoint_grid_json])
        all_cmds.append((job_id, cmd))

    procs: List[Tuple[str, subprocess.Popen]] = []
    for job_id, cmd in all_cmds:
        procs.append((job_id, subprocess.Popen(cmd)))

    total = len(procs)
    failed: List[Tuple[str, int]] = []
    done = 0
    for job_id, proc in procs:
        return_code = proc.wait()
        done += 1
        report(f"progress : {100.0 * done / total:.2f}%")
        if return_code != 0:
            failed.append((job_id, return_code))

    if failed:
        failed_str = ", ".join([f"{job_id}(rc={rc})" for job_id, rc in failed])
        report(f"qs_decentralized: failed jobs: {failed_str}")
        elapsed_ms = int((time.time() - start_time) * 1000.0)
        report(f"compuattion time: {elapsed_ms} ms")
        if report_should_close:
            report_file.close()
        return 1
    elapsed_ms = int((time.time() - start_time) * 1000.0)
    report(f"compuattion time: {elapsed_ms} ms")
    if report_should_close:
        report_file.close()
    return 0


if __name__ == "__main__":
	sys.exit(main())
