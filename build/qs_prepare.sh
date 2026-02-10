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
  echo "qs_prepare: missing job id (-id or -r)" >&2
  exit 1
fi

data_dir="/tmp/G2S/data"
mkdir -p "$data_dir"

link_one() {
  local src_hash="$1"
  local dest_base="$2"
  if [[ -f "$data_dir/${src_hash}.bgrid.gz" ]]; then
    ln -sf "$data_dir/${src_hash}.bgrid.gz" "$data_dir/${dest_base}.bgrid.gz"
  elif [[ -f "$data_dir/${src_hash}.bgrid" ]]; then
    ln -sf "$data_dir/${src_hash}.bgrid" "$data_dir/${dest_base}.bgrid"
  else
    echo "qs_prepare: missing input ${src_hash}" >&2
  fi
}

if [[ ${#tis[@]} -gt 0 ]]; then
  for i in "${!tis[@]}"; do
    link_one "${tis[$i]}" "input_ti_${i}_${job_id}"
  done
fi

if [[ ${#dis[@]} -gt 0 ]]; then
  if [[ ${#dis[@]} -gt 1 ]]; then
    echo "qs_prepare: multiple -di provided, using the first" >&2
  fi
  link_one "${dis[0]}" "input_di_${job_id}"
fi

echo "qs_prepare: job_id=${job_id} ti=${#tis[@]} di=${#dis[@]}" >&2
