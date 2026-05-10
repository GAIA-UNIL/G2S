#!/usr/bin/env bash
set -euo pipefail

job_id=""
log_path=""
declare -a tis=()
declare -a dis=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r)
      log_path="${2:-}"
      shift 2
      ;;
    -id)
      job_id="${2:-}"
      shift 2
      ;;
    -ti)
      shift
      while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
        tis+=("$1")
        shift
      done
      ;;
    -di)
      shift
      while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
        dis+=("$1")
        shift
      done
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$job_id" && -n "$log_path" ]]; then
  job_id="$(basename "$log_path" .log)"
fi

if [[ -z "$job_id" ]]; then
  echo "qs_prepare: missing job id (-id or -r)." >&2
  exit 2
fi

resolve_data_dir() {
  if [[ -n "${G2S_DATA_DIR:-}" ]]; then
    local explicit="${G2S_DATA_DIR%/}"
    if [[ "$(basename "$explicit")" == "data" ]]; then
      printf "%s" "$explicit"
    else
      printf "%s" "$explicit/data"
    fi
    return 0
  fi
  local user_name="${USER:-${LOGNAME:-}}"
  if [[ -n "$user_name" ]]; then
    local scratch_root="/scratch/${user_name}/G2S"
    if [[ -d "$scratch_root" ]]; then
      printf "%s" "$scratch_root/data"
      return 0
    fi
  fi
  printf "%s" "/tmp/G2S/data"
}

data_dir="$(resolve_data_dir)"
mkdir -p "$data_dir"

link_one() {
  local source_name="$1"
  local target_name="$2"

  if [[ -f "$data_dir/${source_name}.bgrid.gz" ]]; then
    ln -sf "$data_dir/${source_name}.bgrid.gz" "$data_dir/${target_name}.bgrid.gz"
    return 0
  fi
  if [[ -f "$data_dir/${source_name}.bgrid" ]]; then
    ln -sf "$data_dir/${source_name}.bgrid" "$data_dir/${target_name}.bgrid"
    return 0
  fi
  if [[ -f "$source_name" ]]; then
    case "$source_name" in
      *.bgrid.gz) ln -sf "$source_name" "$data_dir/${target_name}.bgrid.gz" ;;
      *.bgrid) ln -sf "$source_name" "$data_dir/${target_name}.bgrid" ;;
      *) ln -sf "$source_name" "$data_dir/${target_name}.bgrid" ;;
    esac
    return 0
  fi

  echo "qs_prepare: missing source '${source_name}'." >&2
  return 1
}

missing_count=0

if [[ ${#tis[@]} -gt 0 ]]; then
  for i in "${!tis[@]}"; do
    if ! link_one "${tis[$i]}" "input_ti_${i}_${job_id}"; then
      missing_count=$((missing_count + 1))
    fi
  done
fi

if [[ ${#dis[@]} -gt 0 ]]; then
  if [[ ${#dis[@]} -gt 1 ]]; then
    echo "qs_prepare: multiple -di values, only the first is used." >&2
  fi
  if ! link_one "${dis[0]}" "input_di_${job_id}"; then
    missing_count=$((missing_count + 1))
  fi
fi

if [[ $missing_count -gt 0 ]]; then
  echo "qs_prepare: failed with ${missing_count} missing input(s)." >&2
  exit 1
fi

echo "qs_prepare: job_id=${job_id} ti_count=${#tis[@]} di_count=${#dis[@]}" >&2
