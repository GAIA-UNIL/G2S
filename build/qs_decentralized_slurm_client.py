#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


POLL_INTERVAL_SECONDS = 0.2


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
    out.append(str(node))


def build_like(template: Any, values: List[str], index_ref: List[int]) -> Any:
    if isinstance(template, list):
        return [build_like(child, values, index_ref) for child in template]
    current = values[index_ref[0]]
    index_ref[0] += 1
    return current


def parse_endpoint_port(endpoint: str, default_port: int) -> int:
    stripped = endpoint.strip()
    if ":" not in stripped:
        return default_port
    raw_port = stripped.rsplit(":", 1)[-1]
    try:
        parsed = int(raw_port)
    except Exception:
        return default_port
    if parsed <= 0:
        return default_port
    return parsed


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


def read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_file(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    os.replace(tmp_path, path)


def write_rank_host_file(host_map_dir: str, rank: int, host: str) -> None:
    path = os.path.join(host_map_dir, f"{rank}.json")
    payload = {"rank": rank, "host": host}
    write_json_file(path, payload)


def read_hosts_by_rank(host_map_dir: str, expected_count: int) -> Optional[List[str]]:
    hosts = [""] * expected_count
    for rank in range(expected_count):
        path = os.path.join(host_map_dir, f"{rank}.json")
        if not os.path.isfile(path):
            return None
        payload = read_json_file(path)
        host_value = str(payload.get("host", "")).strip()
        if host_value == "":
            return None
        hosts[rank] = host_value
    return hosts


def wait_for_hosts(host_map_dir: str, expected_count: int, timeout_seconds: int) -> List[str]:
    deadline = time.time() + max(float(timeout_seconds), 1.0)
    while True:
        hosts = read_hosts_by_rank(host_map_dir, expected_count)
        if hosts is not None:
            return hosts
        if time.time() > deadline:
            raise TimeoutError("timeout while waiting for Slurm host map")
        time.sleep(POLL_INTERVAL_SECONDS)


def wait_for_file(path: str, timeout_seconds: int) -> None:
    deadline = time.time() + max(float(timeout_seconds), 1.0)
    while not os.path.isfile(path):
        if time.time() > deadline:
            raise TimeoutError(f"timeout while waiting for file '{path}'")
        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> int:
    entries = parse_arg_entries(sys.argv)
    bundle_file = get_first_value(entries, ("-bundle",))
    if not bundle_file:
        print("qs_dm_slurm_client: missing -bundle", file=sys.stderr)
        return 2

    try:
        bundle = read_json_file(bundle_file)
    except Exception as exc:
        print(f"qs_dm_slurm_client: invalid bundle file ({exc})", file=sys.stderr)
        return 2

    try:
        job_ids = [normalize_job_id(value) for value in bundle["job_ids"]]
    except Exception as exc:
        print(f"qs_dm_slurm_client: invalid bundle job ids ({exc})", file=sys.stderr)
        return 2
    total_jobs = len(job_ids)
    if total_jobs == 0:
        print("qs_dm_slurm_client: bundle contains no jobs", file=sys.stderr)
        return 2

    rank_raw = os.environ.get("SLURM_PROCID", "")
    if rank_raw == "":
        rank_raw = get_first_value(entries, ("-rank",))
    try:
        rank = int(rank_raw)
    except Exception:
        print("qs_dm_slurm_client: missing or invalid rank (SLURM_PROCID)", file=sys.stderr)
        return 2
    if rank < 0 or rank >= total_jobs:
        print(
            f"qs_dm_slurm_client: rank {rank} out of range for {total_jobs} jobs",
            file=sys.stderr,
        )
        return 2

    host = os.environ.get("SLURMD_NODENAME", "").strip()
    if host == "":
        host = socket.gethostname().strip()
    if host == "":
        print("qs_dm_slurm_client: cannot resolve local hostname", file=sys.stderr)
        return 2

    host_map_dir = str(bundle.get("host_map_dir", "")).strip()
    if host_map_dir == "":
        print("qs_dm_slurm_client: missing host_map_dir in bundle", file=sys.stderr)
        return 2
    os.makedirs(host_map_dir, exist_ok=True)

    timeout_seconds = int(bundle.get("coord_timeout_seconds", 1800))
    base_port_start = int(bundle.get("base_port_start", 8130))
    endpoint_placeholder_file = str(bundle.get("endpoint_grid_placeholder_file", ""))
    endpoint_resolved_file = str(bundle.get("endpoint_grid_resolved_file", ""))
    job_grid_file = str(bundle.get("job_grid_file", ""))
    di_grid_file = str(bundle.get("di_grid_file", ""))
    data_dir = str(bundle.get("data_dir", ""))
    log_dir = str(bundle.get("log_dir", ""))
    common_seed = str(bundle.get("common_seed", ""))
    forwarded_args = [str(value) for value in bundle.get("forwarded_args", [])]
    local_ti = [str(value) for value in bundle.get("local_ti", [])]
    local_di = [str(value) for value in bundle.get("local_di", [])]

    if (
        endpoint_placeholder_file == ""
        or endpoint_resolved_file == ""
        or job_grid_file == ""
        or di_grid_file == ""
        or data_dir == ""
        or log_dir == ""
    ):
        print("qs_dm_slurm_client: incomplete bundle payload", file=sys.stderr)
        return 2

    try:
        write_rank_host_file(host_map_dir, rank, host)
        hosts_by_rank = wait_for_hosts(host_map_dir, total_jobs, timeout_seconds)

        if rank == 0:
            endpoint_template = read_json_file(endpoint_placeholder_file)
            flattened_template: List[str] = []
            flatten_row_major(endpoint_template, flattened_template)
            if len(flattened_template) != total_jobs:
                raise RuntimeError("endpoint template size mismatch")

            resolved_flat: List[str] = []
            for index, template_value in enumerate(flattened_template):
                port = parse_endpoint_port(template_value, base_port_start + index)
                resolved_flat.append(f"{hosts_by_rank[index]}:{port}")
            resolved_grid = build_like(endpoint_template, resolved_flat, [0])
            write_json_file(endpoint_resolved_file, resolved_grid)

        wait_for_file(endpoint_resolved_file, timeout_seconds)

        job_id = job_ids[rank]
        ti_values, di_values = ensure_job_inputs(data_dir, job_id, local_ti, local_di)
        command: List[str] = ["./qs", "-r", f"{log_dir}/{job_id}.log", "-s", common_seed, "-ti"]
        command.extend(ti_values)
        command.extend(["-di", di_values[0]])
        command.extend(forwarded_args)
        command.extend(
            ["-jg", job_grid_file, "-eg", endpoint_resolved_file, "-di_grid_json", di_grid_file]
        )
    except Exception as exc:
        print(f"qs_dm_slurm_client: setup failed ({exc})", file=sys.stderr)
        return 1

    try:
        process = subprocess.Popen(command)
        return process.wait()
    except Exception as exc:
        print(f"qs_dm_slurm_client: failed to launch qs ({exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
