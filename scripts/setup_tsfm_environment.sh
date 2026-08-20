#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: setup_tsfm_environment.sh [--dry-run] ENVIRONMENT_KEY ABSOLUTE_EMPTY_TARGET" >&2
  exit 64
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
[[ "$#" -eq 2 ]] || usage

environment_key=$1
target=$2

case "$environment_key" in
  timesfm_v1) python_command=python3.11; python_version=3.11 ;;
  uni2ts) python_command=python3.11; python_version=3.11 ;;
  lag_llama) python_command=python3.10; python_version=3.10 ;;
  granite_tsfm) python_command=python3.11; python_version=3.11 ;;
  timer_legacy) python_command=python3.10; python_version=3.10 ;;
  transformers_recent) python_command=python3.11; python_version=3.11 ;;
  tempo_legacy) python_command=python3.10; python_version=3.10 ;;
  toto2) python_command=python3.12; python_version=3.12 ;;
  kairos) python_command=python3.11; python_version=3.11 ;;
  tirex) python_command=python3.11; python_version=3.11 ;;
  tabpfn_ts) python_command=python3.11; python_version=3.11 ;;
  *)
    echo "error: environment key is not allowlisted" >&2
    exit 65
    ;;
esac

[[ -n "$target" ]] || usage
case "$target" in
  /*) ;;
  *)
    echo "error: target must be an explicit absolute directory" >&2
    exit 66
    ;;
esac
case "$target" in
  *//*|*/./*|*/../*|*/.|*/..)
    echo "error: unsafe target path syntax" >&2
    exit 66
    ;;
esac
if [[ "$target" != "/" ]]; then
  target=${target%/}
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
user_home=${HOME:-}
case "$target" in
  /|"$repository_root"|"$script_dir"|"$user_home")
    echo "error: unsafe target directory" >&2
    exit 66
    ;;
esac

probe=$target
while [[ "$probe" != "/" ]]; do
  if [[ -L "$probe" ]]; then
    echo "error: target must not be or traverse a symbolic link" >&2
    exit 66
  fi
  probe=$(dirname -- "$probe")
done

requirements="$repository_root/configs/tsfm-environments/$environment_key.txt"
vcs_requirements="$repository_root/configs/tsfm-environments/$environment_key.vcs"
lock_validator="$script_dir/validate_tsfm_locks.py"
target_contract="$script_dir/tsfm_target_contract.py"
[[ -f "$requirements" && -f "$lock_validator" && -f "$target_contract" ]] || {
  echo "error: allowlisted environment specification is missing" >&2
  exit 65
}
validator_python=$(command -v python3) || {
  echo "error: Python is required for offline lock validation" >&2
  exit 67
}
clean_path=/usr/local/bin:/usr/bin:/bin
if ! env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
  "$validator_python" -I "$lock_validator" --environment "$environment_key" \
  >/dev/null 2>&1; then
  echo "error: environment lock validation failed" >&2
  exit 68
fi

check_contract() {
  local action=$1
  shift
  local status=0
  env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    "$validator_python" -I "$target_contract" "$action" "$target" "$@" \
    >/dev/null 2>&1 || status=$?
  case "$status" in
    0) return 0 ;;
    2)
      echo "error: target parent is not trusted" >&2
      return 66
      ;;
    *)
      echo "error: target must be an empty trusted directory" >&2
      return 66
      ;;
  esac
}

check_contract check || exit $?

if [[ "$dry_run" -eq 1 ]]; then
  echo "python3 <REPOSITORY>/scripts/validate_tsfm_locks.py --environment $environment_key"
  echo "$python_command -m venv <TARGET>"
  echo "python3 <REPOSITORY>/scripts/validate_tsfm_locks.py --environment $environment_key --emit-build-lock > <BUILD-LOCK>"
  echo "<TARGET>/bin/python -m pip install --require-hashes --only-binary=:all: --requirement <BUILD-LOCK>"
  echo "<TARGET>/bin/python -m pip install --require-hashes --no-build-isolation --requirement <REPOSITORY>/configs/tsfm-environments/$environment_key.txt"
  if [[ -f "$vcs_requirements" ]]; then
    echo "<TARGET>/bin/python -m pip install --no-deps --no-build-isolation --requirement <REPOSITORY>/configs/tsfm-environments/$environment_key.vcs"
  fi
  echo "<TARGET>/bin/python -m pip install --no-deps --no-build-isolation <REPOSITORY>"
  exit 0
fi

if [[ "$(/usr/bin/uname -s 2>/dev/null)" != "Linux" || \
      "$(/usr/bin/uname -m 2>/dev/null)" != "x86_64" ]]; then
  echo "error: environment locks support only Linux x86_64" >&2
  exit 67
fi

python_path=$(command -v "$python_command") || {
  echo "error: required Python interpreter is unavailable" >&2
  exit 67
}
actual_python=$(
  env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    "$python_path" -I -c \
    'import sys; print(f"{sys.implementation.name}:{sys.version_info.major}.{sys.version_info.minor}")' \
    2>/dev/null
) || {
  echo "error: required Python interpreter could not be validated" >&2
  exit 67
}
[[ "$actual_python" == "cpython:$python_version" ]] || {
  echo "error: environment lock requires the declared CPython minor" >&2
  exit 67
}

target_identity=$(
  env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    "$validator_python" -I "$target_contract" prepare "$target" 2>/dev/null
) || {
  echo "error: target must be an empty trusted directory" >&2
  exit 66
}

verify_identity() {
  if ! env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    "$validator_python" -I "$target_contract" verify "$target" "$target_identity" \
    >/dev/null 2>&1; then
    echo "error: environment target identity changed" >&2
    exit 66
  fi
}

run_stage() {
  local stage=$1
  shift
  if "$@" >/dev/null 2>&1; then
    return 0
  fi
  echo "error: $stage failed; subprocess output withheld" >&2
  return 1
}

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN TABPFN_TOKEN NA_ACCEPT_MODEL_LICENSES

verify_identity
run_stage "environment creation" \
  env -i PATH="$clean_path" LANG=C.UTF-8 HOME="$target" TMPDIR="$target" \
  "$python_path" -m venv "$target"
verify_identity
if ! cd -- "$target" 2>/dev/null; then
  echo "error: environment target identity changed" >&2
  exit 66
fi

build_lock=$(mktemp "$target/.na-tsfm-build-lock.XXXXXX" 2>/dev/null) || {
  echo "error: could not prepare hashed build requirements" >&2
  exit 1
}
cleanup_build_lock() {
  rm -f -- "$build_lock" 2>/dev/null || :
}
trap cleanup_build_lock EXIT HUP INT TERM
if ! env -i PATH="$clean_path" LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
  "$validator_python" -I "$lock_validator" --environment "$environment_key" \
  --emit-build-lock >"$build_lock" 2>/dev/null; then
  echo "error: could not prepare hashed build requirements" >&2
  exit 1
fi

verify_identity
run_stage "build requirement installation" \
  env -i PATH="$target/bin:$clean_path" LANG=C.UTF-8 HOME="$target" \
  TMPDIR="$target" PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 \
  "$target/bin/python" -m pip install --require-hashes --only-binary=:all: \
  --requirement "$build_lock"
verify_identity

run_stage "dependency installation" \
  env -i PATH="$target/bin:$clean_path" LANG=C.UTF-8 HOME="$target" \
  TMPDIR="$target" PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 \
  "$target/bin/python" -m pip install --require-hashes --no-build-isolation \
  --requirement "$requirements"
verify_identity

if [[ -f "$vcs_requirements" ]]; then
  run_stage "immutable VCS root installation" \
    env -i PATH="$target/bin:$clean_path" LANG=C.UTF-8 HOME="$target" \
    TMPDIR="$target" PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 \
    "$target/bin/python" -m pip install --no-deps --no-build-isolation \
    --requirement "$vcs_requirements"
  verify_identity
fi

run_stage "repository installation" \
  env -i PATH="$target/bin:$clean_path" LANG=C.UTF-8 HOME="$target" \
  TMPDIR="$target" PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 \
  "$target/bin/python" -m pip install --no-deps --no-build-isolation \
  "$repository_root"
verify_identity
