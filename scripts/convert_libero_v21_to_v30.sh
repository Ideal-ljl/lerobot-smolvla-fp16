#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$PWD}"
dataset_name="libero_goal_no_noops_1.0.0_lerobot"
dataset_path="${repo_root}/${dataset_name}"
backup_path="${repo_root}/${dataset_name}_v21_backup"

if [[ ! -d "${dataset_path}" ]]; then
  echo "Missing v2.1 dataset: ${dataset_path}" >&2
  exit 1
fi
if [[ -e "${backup_path}" ]]; then
  echo "Backup already exists: ${backup_path}" >&2
  exit 1
fi

cp -a "${dataset_path}" "${backup_path}"

PYTHONPATH="${repo_root}/src" python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id="${dataset_name}" \
  --root="${repo_root}" \
  --push-to-hub=false

echo "Converted dataset is at ${dataset_path}"
echo "Original v2.1 backup is at ${backup_path}"
