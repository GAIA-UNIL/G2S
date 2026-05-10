#!/usr/bin/env python3
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple


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
DEFAULT_BASE_PORT = 8130
DEFAULT_NTASKS_PER_NODE = 1
DEFAULT_COORD_TIMEOUT_SECONDS = 1800


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


def parse_positive_int(raw: str, key_name: str) -> int:
    try:
        value = int(float(raw))
    except Exception as exc:
        raise ValueError(f"invalid {key_name} value '{raw}' ({exc})")
    if value <= 0:
        raise ValueError(f"{key_name} must be > 0")
    return value


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


def data_name_exists(data_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(data_dir, f"{name}.bgrid.gz")) or os.path.exists(
        os.path.join(data_dir, f"{name}.bgrid")
    )


def data_reference_exists(data_dir: str, reference: str) -> bool:
    if reference == "":
        return False
    if os.path.exists(reference):
        return True
    return data_name_exists(data_dir, reference)


def get_default_ti(data_dir: str, job_id: str) -> List[str]:
    values: List[str] = []
    index = 0
    while True:
        name = f"input_ti_{index}_{job_id}"
        if not data_name_exists(data_dir, name):
            break
        values.append(name)
        index += 1
    return values


def get_default_di(data_dir: str, job_id: str) -> str:
    name = f"input_di_{job_id}"
    if data_name_exists(data_dir, name):
        return name
    return ""


def default_di_name(job_id: str) -> str:
    return f"input_di_{job_id}"


def ensure_job_inputs(
    data_dir: str, job_id: str, local_ti: List[str], local_di: List[str]
) -> Tuple[List[str], List[str]]:
    if local_ti:
        for value in local_ti:
            if not data_reference_exists(data_dir, value):
                raise RuntimeError(f"-ti value '{value}' does not exist")
        ti_values = local_ti
    else:
        ti_values = get_default_ti(data_dir, job_id)

    if local_di:
        di_value = local_di[0]
        if not data_reference_exists(data_dir, di_value):
            raise RuntimeError(f"-di value '{di_value}' does not exist")
        di_values = [di_value]
    else:
        default_di = get_default_di(data_dir, job_id)
        di_values = [default_di] if default_di else []

    if not ti_values:
        raise RuntimeError(
            f"no TI for job {job_id} (missing -ti and missing input_ti_<index>_{job_id})"
        )
    if not di_values:
        raise RuntimeError(f"no DI for job {job_id} (missing -di and missing input_di_{job_id})")
    return ti_values, di_values


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
        "-slurm_nodes",
        "-slurm_ntasks_per_node",
        "-slurm_account",
        "-account",
        "-slurm_partition",
        "-slurm_qos",
        "-slurm_time",
        "-slurm_base_port",
        "-slurm_coord_timeout",
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


def read_json_payload(payload: str) -> Any:
    if payload and os.path.isfile(payload):
        with open(payload, "r", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(payload)


def write_json_file(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    os.replace(tmp_path, path)


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        stream.write(content)
    os.replace(tmp_path, path)


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

    if shutil.which("sbatch") is None:
        report("qs_dm_slurm: missing 'sbatch' in PATH.")
        if report_should_close:
            report_file.close()
        return 2
    if shutil.which("srun") is None:
        report("qs_dm_slurm: missing 'srun' in PATH.")
        if report_should_close:
            report_file.close()
        return 2

    job_grid_payload = get_first_value(entries, ("-jg", "-job_grid_json", "-job_grid"))
    if not job_grid_payload:
        report("qs_dm_slurm: missing -jg (or -job_grid_json/-job_grid).")
        if report_should_close:
            report_file.close()
        return 2

    try:
        job_grid = read_json_payload(job_grid_payload)
    except Exception as exc:
        report(f"qs_dm_slurm: invalid job grid payload ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    job_ids: List[str] = []
    try:
        flatten_row_major(job_grid, job_ids)
    except Exception as exc:
        report(f"qs_dm_slurm: invalid job identifier in job grid ({exc}).")
        if report_should_close:
            report_file.close()
        return 2
    if not job_ids:
        report("qs_dm_slurm: empty job grid.")
        if report_should_close:
            report_file.close()
        return 2

    if get_first_value(entries, ("-eg", "-endpoint_grid_json")):
        report("qs_dm_slurm: -eg/-endpoint_grid_json is ignored (generated at runtime from Slurm nodes).")

    local_ti = get_all_values(entries, "-ti")
    local_di = get_all_values(entries, "-di")
    if len(local_di) > 1:
        report("qs_dm_slurm: multiple -di values received, only the first will be used.")
        local_di = [local_di[0]]

    forwarded = to_cmd_args(entries)
    common_seed = resolve_seed(entries)

    nodes_raw = get_first_value(entries, ("-slurm_nodes",))
    ntasks_per_node_raw = get_first_value(entries, ("-slurm_ntasks_per_node",))
    account = get_first_value(entries, ("-slurm_account", "-account"))
    partition = get_first_value(entries, ("-slurm_partition",))
    qos = get_first_value(entries, ("-slurm_qos",))
    walltime = get_first_value(entries, ("-slurm_time",))
    base_port_raw = get_first_value(entries, ("-slurm_base_port",))
    timeout_raw = get_first_value(entries, ("-slurm_coord_timeout",))

    try:
        ntasks_per_node = (
            parse_positive_int(ntasks_per_node_raw, "-slurm_ntasks_per_node")
            if ntasks_per_node_raw
            else DEFAULT_NTASKS_PER_NODE
        )
        base_port = (
            parse_positive_int(base_port_raw, "-slurm_base_port")
            if base_port_raw
            else DEFAULT_BASE_PORT
        )
        coord_timeout_seconds = (
            parse_positive_int(timeout_raw, "-slurm_coord_timeout")
            if timeout_raw
            else DEFAULT_COORD_TIMEOUT_SECONDS
        )
        requested_nodes = parse_positive_int(nodes_raw, "-slurm_nodes") if nodes_raw else 0
    except Exception as exc:
        report(f"qs_dm_slurm: invalid slurm option ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if DATA_DIR.startswith("/tmp/") or DATA_DIR == "/tmp":
        report(
            "qs_dm_slurm: warning: G2S_DATA_DIR points to /tmp; "
            "ensure this path is shared across login and compute nodes."
        )

    di_name_flat: List[str] = []
    try:
        for job_id in job_ids:
            ensure_job_inputs(DATA_DIR, job_id, local_ti, local_di)
            di_name_flat.append(local_di[0] if local_di else default_di_name(job_id))
    except Exception as exc:
        report(f"qs_dm_slurm: cannot prepare jobs ({exc}).")
        if report_should_close:
            report_file.close()
        return 2

    di_grid = build_like(job_grid, di_name_flat, [0])
    total_jobs = len(job_ids)
    ntasks = total_jobs
    nodes = requested_nodes if requested_nodes > 0 else int(math.ceil(ntasks / float(ntasks_per_node)))
    if nodes * ntasks_per_node < ntasks:
        report(
            "qs_dm_slurm: invalid node/task layout "
            f"(nodes={nodes}, ntasks_per_node={ntasks_per_node}, ntasks={ntasks})."
        )
        if report_should_close:
            report_file.close()
        return 2

    bundle_id = f"{int(start_time * 1000000)}_{os.getpid()}"
    runtime_dir = os.path.join(DATA_DIR, f"qs_dm_slurm_bundle_{bundle_id}")
    host_map_dir = os.path.join(runtime_dir, "host_map")
    os.makedirs(host_map_dir, exist_ok=True)

    job_grid_file = os.path.join(runtime_dir, "job_grid.json")
    endpoint_grid_placeholder_file = os.path.join(runtime_dir, "endpoint_grid_placeholder.json")
    endpoint_grid_resolved_file = os.path.join(runtime_dir, "endpoint_grid_resolved.json")
    di_grid_file = os.path.join(runtime_dir, "di_grid.json")
    bundle_file = os.path.join(runtime_dir, "bundle.json")
    batch_file = os.path.join(runtime_dir, "submit.sbatch")
    slurm_stdout_file = os.path.join(LOG_DIR, f"qs_dm_slurm_{bundle_id}.out")
    slurm_stderr_file = os.path.join(LOG_DIR, f"qs_dm_slurm_{bundle_id}.err")

    endpoint_placeholder_flat = [f"none:{base_port + index}" for index in range(total_jobs)]
    endpoint_placeholder_grid = build_like(job_grid, endpoint_placeholder_flat, [0])

    write_json_file(job_grid_file, job_grid)
    write_json_file(endpoint_grid_placeholder_file, endpoint_placeholder_grid)
    write_json_file(di_grid_file, di_grid)

    workdir = os.getcwd()
    client_script_path = os.path.join(workdir, "qs_decentralized_slurm_client.py")
    if not os.path.isfile(client_script_path):
        report(f"qs_dm_slurm: missing client script '{client_script_path}'.")
        if report_should_close:
            report_file.close()
        return 2

    bundle_payload: Dict[str, Any] = {
        "bundle_id": bundle_id,
        "job_ids": job_ids,
        "job_grid_file": job_grid_file,
        "endpoint_grid_placeholder_file": endpoint_grid_placeholder_file,
        "endpoint_grid_resolved_file": endpoint_grid_resolved_file,
        "di_grid_file": di_grid_file,
        "host_map_dir": host_map_dir,
        "data_dir": DATA_DIR,
        "log_dir": LOG_DIR,
        "forwarded_args": forwarded,
        "common_seed": common_seed,
        "local_ti": local_ti,
        "local_di": local_di,
        "base_port_start": base_port,
        "coord_timeout_seconds": coord_timeout_seconds,
    }
    write_json_file(bundle_file, bundle_payload)

    batch_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(workdir)}",
        (
            "srun --kill-on-bad-exit=1 "
            f"--ntasks {ntasks} --nodes {nodes} --ntasks-per-node {ntasks_per_node} "
            f"{shlex.quote(client_script_path)} -bundle {shlex.quote(bundle_file)}"
        ),
    ]
    write_text_file(batch_file, "\n".join(batch_lines) + "\n")
    os.chmod(batch_file, 0o755)

    sbatch_cmd: List[str] = [
        "sbatch",
        "--wait",
        "--parsable",
        "--job-name",
        f"qs_dm_slurm_{bundle_id}",
        "--output",
        slurm_stdout_file,
        "--error",
        slurm_stderr_file,
        "--chdir",
        workdir,
        "--ntasks",
        str(ntasks),
        "--nodes",
        str(nodes),
        "--ntasks-per-node",
        str(ntasks_per_node),
    ]
    if account:
        sbatch_cmd.extend(["--account", account])
    if partition:
        sbatch_cmd.extend(["--partition", partition])
    if qos:
        sbatch_cmd.extend(["--qos", qos])
    if walltime:
        sbatch_cmd.extend(["--time", walltime])
    sbatch_cmd.append(batch_file)

    report("qs_dm_slurm: submitting Slurm job...")
    report(" ".join([shlex.quote(token) for token in sbatch_cmd]))
    completed = subprocess.run(
        sbatch_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        report(f"qs_dm_slurm: sbatch output: {stdout}")
    if stderr:
        report(f"qs_dm_slurm: sbatch stderr: {stderr}")

    elapsed_seconds = time.time() - start_time
    elapsed_ms = int(elapsed_seconds * 1000.0)
    if completed.returncode != 0:
        report(f"qs_dm_slurm: Slurm execution failed with return code {completed.returncode}.")
        report(f"compuattion time: {elapsed_seconds:7.2f} s")
        report(f"compuattion time: {elapsed_ms} ms")
        if report_should_close:
            report_file.close()
        return 1

    report("progress : 100.00%")
    report(f"compuattion time: {elapsed_seconds:7.2f} s")
    report(f"compuattion time: {elapsed_ms} ms")
    if report_should_close:
        report_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
