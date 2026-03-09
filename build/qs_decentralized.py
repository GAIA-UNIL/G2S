#!/usr/bin/env python3
import json
import os
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def resolve_storage_paths() -> Tuple[str, str]:
    explicit = os.environ.get("G2S_DATA_DIR", "").strip()
    if explicit:
        normalized = os.path.normpath(explicit)
        if os.path.basename(normalized) == "data":
            root_dir = os.path.dirname(normalized)
            data_dir = normalized
        else:
            root_dir = normalized
            data_dir = os.path.join(root_dir, "data")
        return data_dir, os.path.join(root_dir, "logs")

    user_name = os.environ.get("USER", "").strip() or os.environ.get("LOGNAME", "").strip()
    if user_name:
        scratch_root = f"/scratch/{user_name}/G2S"
        if os.path.isdir(scratch_root):
            return os.path.join(scratch_root, "data"), os.path.join(scratch_root, "logs")
    return "/tmp/G2S/data", "/tmp/G2S/logs"


DATA_DIR, LOG_DIR = resolve_storage_paths()
DEFAULT_HOST = "localhost"
DEFAULT_PORT_START = 8130
PROGRESS_PATTERN = re.compile(r"progress\s*:\s*([0-9]+(?:\.[0-9]+)?)%")
PROGRESS_POLL_INTERVAL_SECONDS = 0.2
PROGRESS_REPORT_INTERVAL_SECONDS = 1.0


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
    values: List[str] = []
    for current_key, current_values in entries:
        if current_key == key:
            values.extend(current_values)
    return values


def normalize_job_id(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError("empty job id")
        parsed = int(stripped)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError("non integer job id")
        parsed = int(value)
    else:
        raise ValueError("unsupported job id type")

    if parsed < 0:
        raise ValueError("negative job id")
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
    current = values[index_ref[0]]
    index_ref[0] += 1
    return current


def data_name_exists(name: str) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, f"{name}.bgrid.gz")) or os.path.exists(
        os.path.join(DATA_DIR, f"{name}.bgrid")
    )


def data_reference_exists(reference: str) -> bool:
    if reference == "":
        return False
    if os.path.exists(reference):
        return True
    return data_name_exists(reference)


def get_default_ti(job_id: str) -> List[str]:
    values: List[str] = []
    index = 0
    while True:
        name = f"input_ti_{index}_{job_id}"
        if not data_name_exists(name):
            break
        values.append(name)
        index += 1
    return values


def get_default_di(job_id: str) -> str:
    name = f"input_di_{job_id}"
    if data_name_exists(name):
        return name
    return ""


def default_di_name(job_id: str) -> str:
    return f"input_di_{job_id}"


def should_skip_forwarded_arg(key: str) -> bool:
    consumed = {
        "-a",
        "-id",
        "-r",
        "-s",
        "-job_grid_json",
        "-job_grid",
        "-jg",
        "-eg",
        "-endpoint_grid_json",
        "-di_grid_json",
        "-ti",
        "-di",
    }
    return key in consumed


def to_cmd_args(entries: Sequence[Tuple[str, List[str]]]) -> List[str]:
    args: List[str] = []
    for key, values in entries:
        if should_skip_forwarded_arg(key):
            continue
        args.append(key)
        args.extend(values)
    return args


def resolve_seed(entries: Sequence[Tuple[str, List[str]]]) -> str:
    seed_value = get_first_value(entries, ("-s",))
    if seed_value == "":
        return str(random.randint(1, 2**31 - 1))
    parsed = int(float(seed_value))
    return str(parsed)


def ensure_job_inputs(job_id: str, local_ti: List[str], local_di: List[str]) -> Tuple[List[str], List[str]]:
    if local_ti:
        for value in local_ti:
            if not data_reference_exists(value):
                raise RuntimeError(f"-ti value '{value}' does not exist")
        ti_values = local_ti
    else:
        ti_values = get_default_ti(job_id)

    if local_di:
        di_value = local_di[0]
        if not data_reference_exists(di_value):
            raise RuntimeError(f"-di value '{di_value}' does not exist")
        di_values = [di_value]
    else:
        default_di = get_default_di(job_id)
        di_values = [default_di] if default_di else []

    if not ti_values:
        raise RuntimeError(
            f"no TI for job {job_id} (missing -ti and missing input_ti_<index>_{job_id})"
        )
    if not di_values:
        raise RuntimeError(f"no DI for job {job_id} (missing -di and missing input_di_{job_id})")
    return ti_values, di_values


def read_json_payload(payload: str) -> Any:
    if payload and os.path.isfile(payload):
        with open(payload, "r", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(payload)


def write_json_file(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))


def parse_progress_from_line(line: str) -> Optional[float]:
    match = PROGRESS_PATTERN.search(line)
    if match is None:
        return None
    value = float(match.group(1))
    if value < 0.0:
        return 0.0
    if value > 100.0:
        return 100.0
    return value


def read_progress_from_log(
    log_path: str, offset: int, partial_line: str, finalize: bool = False
) -> Tuple[int, str, Optional[float]]:
    latest_progress: Optional[float] = None
    chunk = b""
    new_offset = offset
    try:
        with open(log_path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            file_size = stream.tell()
            if file_size < offset:
                offset = 0
                partial_line = ""
            if file_size > offset:
                stream.seek(offset)
                chunk = stream.read(file_size - offset)
                new_offset = offset + len(chunk)
            else:
                new_offset = file_size
    except OSError:
        if finalize and partial_line:
            return new_offset, "", parse_progress_from_line(partial_line.strip())
        return new_offset, partial_line, None

    text = partial_line + chunk.decode("utf-8", errors="ignore")
    if text.endswith("\n"):
        lines = text.splitlines()
        remaining = ""
    else:
        pieces = text.split("\n")
        lines = pieces[:-1]
        remaining = pieces[-1]

    for line in lines:
        progress = parse_progress_from_line(line.strip())
        if progress is not None:
            latest_progress = progress

    if finalize and remaining:
        progress = parse_progress_from_line(remaining.strip())
        if progress is not None:
            latest_progress = progress
        remaining = ""

    return new_offset, remaining, latest_progress


def main() -> int:
    start_time = time.time()
    entries = parse_arg_entries(sys.argv)

    report_path = get_first_value(entries, ("-r",))
    report_file = sys.stderr
    report_should_close = False
    if report_path == "stdout":
        report_file = sys.stdout
    elif report_path and report_path != "stderr":
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        report_file = open(report_path, "a", buffering=1)
        report_should_close = True

    def report(message: str) -> None:
        report_file.write(message + "\n")

    report(" ".join(sys.argv))

    job_grid_payload = get_first_value(entries, ("-jg", "-job_grid_json", "-job_grid"))
    if not job_grid_payload:
        report("qs_dm: missing -jg (or -job_grid_json/-job_grid).")
        if report_should_close:
            report_file.close()
        return 2

    try:
        job_grid = read_json_payload(job_grid_payload)
    except Exception as exc:
        report(f"qs_dm: invalid job grid payload ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    job_ids: List[str] = []
    try:
        flatten_row_major(job_grid, job_ids)
    except Exception as exc:
        report(f"qs_dm: invalid job identifier in job grid ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    if not job_ids:
        report("qs_dm: empty job grid.")
        if report_should_close:
            report_file.close()
        return 2

    endpoints = [f"{DEFAULT_HOST}:{DEFAULT_PORT_START + index}" for index in range(len(job_ids))]
    endpoint_grid = build_like(job_grid, endpoints, [0])

    local_ti = get_all_values(entries, "-ti")
    local_di = get_all_values(entries, "-di")
    if len(local_di) > 1:
        report("qs_dm: multiple -di values received, only the first will be used.")
        local_di = [local_di[0]]

    forwarded = to_cmd_args(entries)
    common_seed = resolve_seed(entries)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_cmds: List[Tuple[str, str, List[str]]] = []
    di_name_flat: List[str] = []
    try:
        for job_id in job_ids:
            ti_values, di_values = ensure_job_inputs(job_id, local_ti, local_di)
            di_value = di_values[0]

            di_name_for_grid = local_di[0] if local_di else default_di_name(job_id)
            di_name_flat.append(di_name_for_grid)

            log_path = os.path.join(LOG_DIR, f"{job_id}.log")
            command: List[str] = ["./qs", "-r", log_path, "-s", common_seed, "-ti"]
            command.extend(ti_values)
            command.extend(["-di", di_value])
            command.extend(forwarded)
            all_cmds.append((job_id, log_path, command))
    except Exception as exc:
        report(f"qs_dm: cannot prepare jobs ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    di_grid = build_like(job_grid, di_name_flat, [0])

    bundle_id = f"{int(start_time * 1000000)}_{os.getpid()}"
    job_grid_file = os.path.join(DATA_DIR, f"qs_dm_job_grid_{bundle_id}.json")
    endpoint_grid_file = os.path.join(DATA_DIR, f"qs_dm_endpoint_grid_{bundle_id}.json")
    di_grid_file = os.path.join(DATA_DIR, f"qs_dm_di_grid_{bundle_id}.json")
    write_json_file(job_grid_file, job_grid)
    write_json_file(endpoint_grid_file, endpoint_grid)
    write_json_file(di_grid_file, di_grid)

    for _, _, command in all_cmds:
        command.extend(["-jg", job_grid_file, "-eg", endpoint_grid_file, "-di_grid_json", di_grid_file])

    processes: List[Tuple[str, str, subprocess.Popen]] = []
    try:
        for job_id, log_path, command in all_cmds:
            processes.append((job_id, log_path, subprocess.Popen(command)))
    except Exception as exc:
        report(f"qs_dm: failed to launch child jobs ({exc}).")
        if report_should_close:
            report_file.close()
        return 1

    total = len(processes)
    job_progress: Dict[str, float] = {}
    log_offsets: Dict[str, int] = {}
    log_fragments: Dict[str, str] = {}
    completed_jobs: Set[str] = set()
    completed = 0
    failed: List[Tuple[str, int]] = []

    for job_id, log_path, _ in processes:
        initial_offset = 0
        try:
            initial_offset = os.path.getsize(log_path)
        except OSError:
            initial_offset = 0
        log_offsets[job_id] = initial_offset
        log_fragments[job_id] = ""
        job_progress[job_id] = 0.0

    def update_job_progress(job_id: str, log_path: str, finalize: bool = False) -> None:
        offset, fragment, progress = read_progress_from_log(
            log_path, log_offsets[job_id], log_fragments[job_id], finalize
        )
        log_offsets[job_id] = offset
        log_fragments[job_id] = fragment
        if progress is not None:
            job_progress[job_id] = max(job_progress[job_id], progress)

    last_progress = -1.0
    last_report_time = 0.0
    while completed < total:
        now = time.time()
        for job_id, log_path, process in processes:
            if job_id in completed_jobs:
                continue

            update_job_progress(job_id, log_path)
            code = process.poll()
            if code is None:
                continue

            update_job_progress(job_id, log_path, finalize=True)
            completed_jobs.add(job_id)
            completed += 1
            job_progress[job_id] = 100.0
            if code != 0:
                failed.append((job_id, code))
            report(f"qs_dm: job {job_id} finished rc={code} ({completed}/{total})")

        aggregate_progress = sum(job_progress.values()) / float(total)
        should_report = (
            (last_progress < 0.0)
            or (aggregate_progress > last_progress + 1e-9)
            or (completed == total and last_progress < 100.0)
        )
        if should_report and (now - last_report_time >= PROGRESS_REPORT_INTERVAL_SECONDS):
            elapsed = max(now - start_time, 0.0)
            if aggregate_progress > 0.0 and aggregate_progress < 100.0:
                eta = elapsed * (100.0 - aggregate_progress) / aggregate_progress
                report(
                    f"progress : {aggregate_progress:.2f}% ({completed}/{total} done, eta ~{eta:7.2f} s)"
                )
            else:
                report(f"progress : {aggregate_progress:.2f}% ({completed}/{total} done)")
            last_progress = aggregate_progress
            last_report_time = now

        if completed < total:
            time.sleep(PROGRESS_POLL_INTERVAL_SECONDS)

    report("progress : 100.00%")
    elapsed_seconds = time.time() - start_time
    elapsed_ms = int(elapsed_seconds * 1000.0)

    if failed:
        failed_jobs = ", ".join([f"{job_id}(rc={code})" for job_id, code in failed])
        report(f"qs_dm: failed jobs: {failed_jobs}")
        report(f"compuattion time: {elapsed_seconds:7.2f} s")
        report(f"compuattion time: {elapsed_ms} ms")
        if report_should_close:
            report_file.close()
        return 1

    report(f"compuattion time: {elapsed_seconds:7.2f} s")
    report(f"compuattion time: {elapsed_ms} ms")
    if report_should_close:
        report_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
