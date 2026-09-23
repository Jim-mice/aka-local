#!/usr/bin/env bash
set -u
workspace="$1"
job_id="$2"
candidate="$3"
mode="$4"
job_dir="$workspace/remote_jobs/$job_id"
package="$workspace/package"
python_bin="$workspace/venv-torch271/bin/python"
selected_uuid="GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d"
write_status() {
  tmp="$job_dir/status.json.tmp"
  printf '{"job_id":"%s","status":"%s","detail":"%s","updated_epoch":%s}\n' "$job_id" "$1" "$2" "$(date +%s)" > "$tmp"
  sync "$tmp"
  mv "$tmp" "$job_dir/status.json"
}
mkdir -p "$job_dir"
write_status "QUEUED" "preflight"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,pci.bus_id,driver_version --format=csv,noheader,nounits > "$job_dir/gpu_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$job_dir/compute_apps_before.csv"
nvidia-smi pmon -c 1 > "$job_dir/pmon_before.txt"
if grep -q "$selected_uuid" "$job_dir/compute_apps_before.csv"; then
  write_status "CONTAMINATED_NOT_RUN" "selected GPU has a compute process"
  exit 42
fi
write_status "RUNNING" "L1/L2 $mode"
export CUDA_VISIBLE_DEVICES="1"
export PATH="$workspace/venv-torch271/bin:$PATH"
export PYTHONPATH="/home/bencheng/aka_targets/megatron-lm-5be9626"
extra=""
if [ "$mode" = "smoke" ]; then extra="--smoke"; fi
"$python_bin" "$package/l1_l2_runner.py" --candidate "$candidate" --output "$job_dir/result.json" $extra > "$job_dir/stdout.log" 2> "$job_dir/stderr.log"
rc=$?
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,pci.bus_id,driver_version --format=csv,noheader,nounits > "$job_dir/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$job_dir/compute_apps_after.csv"
nvidia-smi pmon -c 1 > "$job_dir/pmon_after.txt"
sha256sum "$candidate" "$package/l1_l2_runner.py" "$package/contract_v2_snapshot.json" > "$job_dir/input_hashes.txt"
if [ "$rc" -eq 0 ] && [ -s "$job_dir/result.json" ]; then write_status "DONE" "L1/L2 artifact complete"; else write_status "FAILED" "runner exit $rc"; fi
exit "$rc"
