#!/usr/bin/env bash
set -euo pipefail

readonly CANDIDATE_COMMIT="a3842d1b77bc79e2f70cefcbab136207e7067065"
readonly PYTHON_VERSION="3.11.14"
readonly PATCH_SHA256="0336aa404ce805a160986857763ad89dbe72990d3afe662084a0d08d9c20c366"
readonly CANDIDATE_PYPROJECT_SHA256="105c71eac181e8a9facf97eec448e10760494063f724e6c7b2b403f5ac6483a8"
readonly CANDIDATE_UV_LOCK_SHA256="af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e"
readonly -a DEFAULT_ENVS=(
    PickXtimes
    StopCube
    SwingXtimes
    BinFill
    VideoUnmaskSwap
    VideoUnmask
    ButtonUnmaskSwap
    ButtonUnmask
    VideoRepick
    VideoPlaceButton
    VideoPlaceOrder
    PickHighlight
    InsertPeg
    MoveCube
    PatternLock
    RouteStick
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
ARTIFACTS_ROOT="$WORKSPACE_ROOT/artifacts"
CACHE_ROOT="$WORKSPACE_ROOT/.cache"
WORKTREE="$ARTIFACTS_ROOT/recovery/worktrees/${CANDIDATE_COMMIT}-recovered-clean-v3"
CANDIDATE_VENV="$ARTIFACTS_ROOT/recovery/uv-envs/${CANDIDATE_COMMIT}-py${PYTHON_VERSION}"
PATCH_FILE="$SCRIPT_DIR/generate_dataset_a3842d1.patch"
GENERATOR_REL="scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py"
VALIDATOR="$SCRIPT_DIR/validate_generated_dataset_contract.py"
GENERATED_BASE="$ARTIFACTS_ROOT/generated/$CANDIDATE_COMMIT"

usage() {
    cat <<'EOF'
用法：
  scripts/data-generation/generate_dataset.sh --output-dir <候选专用的新目录> [选项]

默认完整生成：
  16 个环境 × 每环境 100 episodes，20 workers，GPU 1。
  必须显式提供全新（不存在或为空）的仓库内 --output-dir；--help 不需要该参数。

可选参数：
  --env, -e <ENV...>       生成指定环境；可传多个值或逗号分隔。默认全部 16 个环境。
  --episodes, -n <1..100> 每环境 episode 数。默认 100。
  --max-workers, -w <N>   并行 worker 数。默认 20。
  --gpus <0|1|0,1|1,0>   渲染 GPU。默认 1。
  --help, -h              显示本帮助，不创建 worktree 或输出目录。

示例（16×9 审查范围）：
  scripts/data-generation/generate_dataset.sh \
    --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/new-16x9-run \
    --episodes 9 --max-workers 9 --gpus 0,1

安全约束：
  候选 commit、候选 uv.lock、Python 3.11.14、train metadata、原 seed 单次尝试
  以及记录模式均固定。拒绝仓库外/非空输出、seed 递增、关闭 timestep 记录，
  生成后必须通过文件集合、episode、metadata、setup 与完成标志契约检查。
EOF
}

fail() {
    echo "错误：$*" >&2
    exit 2
}

assert_no_workspace_mounts() {
    local parent_mount
    local workspace_mount
    local mount_listing
    local mount_target

    if ! parent_mount="$(findmnt --noheadings --raw --output TARGET --target "$(dirname "$WORKSPACE_ROOT")")"; then
        fail "无法查询 workspace 父目录的 mount 状态。"
    fi
    if ! workspace_mount="$(findmnt --noheadings --raw --output TARGET --target "$WORKSPACE_ROOT")"; then
        fail "无法查询 workspace 的 mount 状态。"
    fi
    if [[ "$workspace_mount" != "$parent_mount" ]]; then
        fail "workspace 本身禁止作为独立 mount/bind mount：$workspace_mount"
    fi
    if ! mount_listing="$(findmnt --submounts --noheadings --raw --output TARGET --target "$WORKSPACE_ROOT")"; then
        fail "无法扫描 workspace 下的 mount。"
    fi
    while IFS= read -r mount_target; do
        case "$mount_target" in
            "$WORKSPACE_ROOT"|"$WORKSPACE_ROOT"/*)
                fail "workspace 内禁止 mount/bind mount：$mount_target"
                ;;
        esac
    done <<< "$mount_listing"
}

assert_safe_repo_directory_path() {
    local candidate="$1"
    local label="$2"
    local lexical
    local resolved
    local relative
    local current="$WORKSPACE_ROOT"
    local part
    local -a parts=()

    if ! lexical="$(realpath -ms -- "$candidate")"; then
        fail "无法规范化 $label：$candidate"
    fi
    case "$lexical" in
        "$WORKSPACE_ROOT"|"$WORKSPACE_ROOT"/*)
            ;;
        *)
            fail "$label 位于仓库外：$lexical"
            ;;
    esac

    relative="${lexical#"$WORKSPACE_ROOT"}"
    if [[ -n "${relative#/}" ]]; then
        IFS='/' read -r -a parts <<< "${relative#/}"
    fi
    for part in "${parts[@]}"; do
        current="$current/$part"
        [[ ! -L "$current" ]] || fail "$label 路径中禁止符号链接：$current"
        if [[ -e "$current" && ! -d "$current" ]]; then
            fail "$label 路径分量不是目录：$current"
        fi
    done

    if ! resolved="$(realpath -m -- "$lexical")"; then
        fail "无法解析 $label：$lexical"
    fi
    if [[ "$resolved" != "$lexical" ]]; then
        fail "$label 解析后发生路径漂移：$lexical -> $resolved"
    fi
}

sanitize_execution_environment() {
    local env_line
    local env_name
    while IFS= read -r env_line; do
        env_name="${env_line%%=*}"
        if [[ "$env_name" =~ ^(UV|GIT)_[A-Za-z0-9_]+$ ]]; then
            unset "$env_name"
        fi
    done < <(env)
    unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
    export GIT_NO_REPLACE_OBJECTS=1
}

verify_worktree_closure() {
    local changed_listing
    local index_entry
    local index_listing
    local abnormal_index_listing=""
    local replace_listing
    local untracked_listing
    local ignored_listing

    if ! replace_listing="$(git -C "$WORKTREE" for-each-ref --format='%(refname)' refs/replace)"; then
        fail "无法扫描恢复 worktree 的 Git replace refs。"
    fi
    [[ -z "$replace_listing" ]] || fail "恢复仓库禁止 Git replace refs：$replace_listing"
    if ! index_listing="$(git -C "$WORKTREE" ls-files -v)"; then
        fail "无法扫描恢复 worktree 的 Git index 标志。"
    fi
    while IFS= read -r index_entry; do
        [[ -n "$index_entry" ]] || continue
        if [[ "${index_entry:0:2}" != "H " ]]; then
            abnormal_index_listing+="${abnormal_index_listing:+$'\n'}$index_entry"
        fi
    done <<< "$index_listing"
    [[ -z "$abnormal_index_listing" ]] || \
        fail "恢复 worktree 禁止 assume-unchanged/skip-worktree 等非标准 index 标志：$abnormal_index_listing"
    if ! git -C "$WORKTREE" diff --cached --quiet; then
        fail "恢复 worktree 存在 staged 修改。"
    fi
    if ! changed_listing="$(git -C "$WORKTREE" diff --name-only HEAD)"; then
        fail "无法读取恢复 worktree 的 tracked diff。"
    fi
    if [[ "$changed_listing" != "$GENERATOR_REL" ]]; then
        fail "恢复 worktree 的 tracked diff 不只包含 $GENERATOR_REL：$changed_listing"
    fi
    if ! untracked_listing="$(git -C "$WORKTREE" ls-files --others --exclude-standard)"; then
        fail "无法扫描恢复 worktree 的未跟踪文件。"
    fi
    [[ -z "$untracked_listing" ]] || fail "恢复 worktree 存在未跟踪文件：$untracked_listing"
    if ! ignored_listing="$(git -C "$WORKTREE" ls-files --others --ignored --exclude-standard)"; then
        fail "无法扫描恢复 worktree 的 ignored 文件。"
    fi
    [[ -z "$ignored_listing" ]] || fail "恢复 worktree 存在 ignored 文件：$ignored_listing"
    if ! cmp -s "$PATCH_FILE" <(git -C "$WORKTREE" diff --no-color --full-index --unified=0 HEAD -- "$GENERATOR_REL"); then
        fail "恢复 worktree 的完整生成器 diff 与固化补丁不一致。"
    fi
    if ! git -C "$WORKTREE" diff --check; then
        fail "恢复 worktree diff 存在空白错误。"
    fi
}

is_supported_env() {
    local candidate="$1"
    local known
    for known in "${DEFAULT_ENVS[@]}"; do
        if [[ "$candidate" == "$known" ]]; then
            return 0
        fi
    done
    return 1
}

append_env_values() {
    local raw="$1"
    local env_id
    local existing
    local -a split_values=()
    IFS=',' read -r -a split_values <<< "$raw"
    for env_id in "${split_values[@]}"; do
        env_id="${env_id#"${env_id%%[![:space:]]*}"}"
        env_id="${env_id%"${env_id##*[![:space:]]}"}"
        [[ -n "$env_id" ]] || fail "--env 包含空环境名。"
        is_supported_env "$env_id" || fail "不支持的环境：$env_id"
        for existing in "${selected_envs[@]}"; do
            [[ "$existing" != "$env_id" ]] || fail "--env 重复指定环境：$env_id"
        done
        selected_envs+=("$env_id")
    done
}

for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        usage
        exit 0
    fi
done

if ! UV_BIN="$(type -P uv)" || [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
    fail "当前 workspace 未找到 uv。"
fi
if ! UV_BIN="$(realpath -e -- "$UV_BIN")"; then
    fail "无法解析 uv 可执行文件。"
fi
readonly UV_BIN
if ! command -v findmnt >/dev/null 2>&1; then
    fail "当前 workspace 未找到用于拒绝 bind mount 的 findmnt。"
fi
if [[ ! -f "$WORKSPACE_ROOT/pyproject.toml" || ! -f "$WORKSPACE_ROOT/uv.lock" ]]; then
    fail "workspace 缺少 pyproject.toml 或 uv.lock。"
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

saw_output_dir=false
saw_env=false
saw_episodes=false
saw_max_workers=false
saw_gpus=false
raw_output_dir=""
episodes=100
max_workers=20
gpus="1"
selected_envs=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace-root|--workspace-root=*|--metadata-root|--metadata-root=*|--max-seed-attempts|--max-seed-attempts=*)
            fail "参数 $1 由恢复入口固定，不能覆盖。"
            ;;
        --seed|--seed=*)
            fail "不能覆盖 metadata 原 seed，也不能启用 seed 递增。"
            ;;
        --no-save-video|--no-save-video=*)
            fail "历史入口的 --no-save-video 同时关闭 timestep 记录，不能用于生成 dataset。"
            ;;
        --save-video)
            shift
            ;;
        --output-dir)
            [[ "$saw_output_dir" == false ]] || fail "--output-dir 只能传入一次。"
            [[ $# -ge 2 ]] || fail "--output-dir 缺少路径。"
            raw_output_dir="$2"
            saw_output_dir=true
            shift 2
            ;;
        --output-dir=*)
            [[ "$saw_output_dir" == false ]] || fail "--output-dir 只能传入一次。"
            raw_output_dir="${1#--output-dir=}"
            saw_output_dir=true
            shift
            ;;
        --env|-e)
            [[ "$saw_env" == false ]] || fail "--env 只能传入一组值。"
            saw_env=true
            shift
            [[ $# -gt 0 && "$1" != -* ]] || fail "--env 至少需要一个环境名。"
            while [[ $# -gt 0 && "$1" != -* ]]; do
                append_env_values "$1"
                shift
            done
            ;;
        --env=*)
            [[ "$saw_env" == false ]] || fail "--env 只能传入一组值。"
            saw_env=true
            append_env_values "${1#--env=}"
            shift
            ;;
        --episodes|-n)
            [[ "$saw_episodes" == false ]] || fail "--episodes 只能传入一次。"
            [[ $# -ge 2 ]] || fail "--episodes 缺少整数值。"
            episodes="$2"
            saw_episodes=true
            shift 2
            ;;
        --episodes=*)
            [[ "$saw_episodes" == false ]] || fail "--episodes 只能传入一次。"
            episodes="${1#--episodes=}"
            saw_episodes=true
            shift
            ;;
        --max-workers|-w)
            [[ "$saw_max_workers" == false ]] || fail "--max-workers 只能传入一次。"
            [[ $# -ge 2 ]] || fail "--max-workers 缺少整数值。"
            max_workers="$2"
            saw_max_workers=true
            shift 2
            ;;
        --max-workers=*)
            [[ "$saw_max_workers" == false ]] || fail "--max-workers 只能传入一次。"
            max_workers="${1#--max-workers=}"
            saw_max_workers=true
            shift
            ;;
        --gpus)
            [[ "$saw_gpus" == false ]] || fail "--gpus 只能传入一次。"
            [[ $# -ge 2 ]] || fail "--gpus 缺少 GPU 列表。"
            gpus="$2"
            saw_gpus=true
            shift 2
            ;;
        --gpus=*)
            [[ "$saw_gpus" == false ]] || fail "--gpus 只能传入一次。"
            gpus="${1#--gpus=}"
            saw_gpus=true
            shift
            ;;
        *)
            fail "不支持的参数 $1；请使用 --help 查看允许的生成参数。"
            ;;
    esac
done

[[ "$saw_output_dir" == true ]] || fail "必须显式传入 --output-dir。"
[[ -n "$raw_output_dir" ]] || fail "--output-dir 不能为空。"
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || fail "--episodes 必须是 1 到 100 的整数。"
(( episodes <= 100 )) || fail "--episodes 不能超过 train metadata 的 100 条记录。"
[[ "$max_workers" =~ ^[1-9][0-9]*$ ]] || fail "--max-workers 必须是正整数。"
gpus="${gpus//[[:space:]]/}"
[[ "$gpus" =~ ^(0|1|0,1|1,0)$ ]] || fail "--gpus 只支持 0、1、0,1 或 1,0。"

if [[ "$saw_env" == false ]]; then
    selected_envs=("${DEFAULT_ENVS[@]}")
fi
[[ ${#selected_envs[@]} -gt 0 ]] || fail "至少需要一个环境。"

if [[ "$raw_output_dir" = /* ]]; then
    output_candidate="$raw_output_dir"
else
    output_candidate="$WORKSPACE_ROOT/$raw_output_dir"
fi
if ! normalized_output_dir="$(realpath -ms -- "$output_candidate")"; then
    fail "无法规范化 --output-dir：$output_candidate"
fi
assert_no_workspace_mounts
assert_safe_repo_directory_path "$ARTIFACTS_ROOT" "artifacts 根目录"
assert_safe_repo_directory_path "$GENERATED_BASE" "候选生成根目录"
assert_safe_repo_directory_path "$normalized_output_dir" "output-dir"
case "$normalized_output_dir" in
    "$GENERATED_BASE"/*)
        ;;
    *)
        fail "--output-dir 必须是 $GENERATED_BASE 的独立子目录；得到 $normalized_output_dir。"
        ;;
esac
if [[ -e "$normalized_output_dir" && ! -d "$normalized_output_dir" ]]; then
    fail "--output-dir 已存在且不是目录：$normalized_output_dir"
fi
if [[ -d "$normalized_output_dir" ]]; then
    if ! first_output_entry="$(find "$normalized_output_dir" -mindepth 1 -print -quit)"; then
        fail "无法扫描 --output-dir 是否为空：$normalized_output_dir"
    fi
    [[ -z "$first_output_entry" ]] || fail "--output-dir 必须不存在或为空，拒绝复用旧生成数据：$normalized_output_dir"
fi

[[ -f "$PATCH_FILE" && ! -L "$PATCH_FILE" ]] || fail "缺少无符号链接的固化恢复补丁：$PATCH_FILE"
[[ -f "$VALIDATOR" && ! -L "$VALIDATOR" ]] || fail "缺少无符号链接的生成后契约验证器：$VALIDATOR"

sanitize_execution_environment
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv-python"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg-cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg-config"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1

assert_safe_repo_directory_path "$CACHE_ROOT" "cache 根目录"
assert_safe_repo_directory_path "$UV_CACHE_DIR" "uv cache"
assert_safe_repo_directory_path "$UV_PYTHON_INSTALL_DIR" "uv Python 安装目录"
assert_safe_repo_directory_path "$XDG_CACHE_HOME" "XDG cache"
assert_safe_repo_directory_path "$XDG_CONFIG_HOME" "XDG config"
assert_safe_repo_directory_path "$TMPDIR" "临时目录"
assert_safe_repo_directory_path "$(dirname "$WORKTREE")" "worktree 根目录"
assert_safe_repo_directory_path "$CANDIDATE_VENV" "候选虚拟环境"
mkdir -p "$(dirname "$WORKTREE")" "$(dirname "$CANDIDATE_VENV")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TMPDIR"
assert_safe_repo_directory_path "$CACHE_ROOT" "cache 根目录"
assert_safe_repo_directory_path "$(dirname "$WORKTREE")" "worktree 根目录"
assert_safe_repo_directory_path "$(dirname "$CANDIDATE_VENV")" "候选虚拟环境根目录"
assert_no_workspace_mounts

if [[ ! -e "$WORKTREE/.git" ]]; then
    git -C "$WORKSPACE_ROOT" worktree add --detach "$WORKTREE" "$CANDIDATE_COMMIT"
fi
assert_safe_repo_directory_path "$WORKTREE" "恢复 worktree"

actual_commit="$(git -C "$WORKTREE" rev-parse HEAD)"
[[ "$actual_commit" == "$CANDIDATE_COMMIT" ]] || \
    fail "恢复 worktree HEAD 为 $actual_commit，预期 $CANDIDATE_COMMIT。"

actual_patch_sha256="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
[[ "$actual_patch_sha256" == "$PATCH_SHA256" ]] || \
    fail "恢复补丁 SHA-256 为 $actual_patch_sha256，预期 $PATCH_SHA256。"

if git -C "$WORKTREE" apply --unidiff-zero --reverse --check "$PATCH_FILE" 2>/dev/null; then
    :
elif git -C "$WORKTREE" apply --unidiff-zero --check "$PATCH_FILE"; then
    git -C "$WORKTREE" apply --unidiff-zero "$PATCH_FILE"
else
    fail "恢复补丁既无法应用，也不是已应用状态；拒绝覆盖现有修改。"
fi

verify_worktree_closure

[[ -f "$WORKTREE/pyproject.toml" && -f "$WORKTREE/uv.lock" ]] || \
    fail "候选 worktree 缺少 pyproject.toml 或 uv.lock。"
actual_pyproject_sha256="$(sha256sum "$WORKTREE/pyproject.toml" | awk '{print $1}')"
actual_uv_lock_sha256="$(sha256sum "$WORKTREE/uv.lock" | awk '{print $1}')"
[[ "$actual_pyproject_sha256" == "$CANDIDATE_PYPROJECT_SHA256" ]] || \
    fail "候选 pyproject.toml SHA-256 漂移：$actual_pyproject_sha256"
[[ "$actual_uv_lock_sha256" == "$CANDIDATE_UV_LOCK_SHA256" ]] || \
    fail "候选 uv.lock SHA-256 漂移：$actual_uv_lock_sha256"

candidate_args=(
    --output-dir "$normalized_output_dir"
    --env "${selected_envs[@]}"
    --episodes "$episodes"
    --max-workers "$max_workers"
    --gpus "$gpus"
)

"$UV_BIN" python install --managed-python --no-config --no-bin --install-dir "$UV_PYTHON_INSTALL_DIR" "$PYTHON_VERSION"
"$UV_BIN" venv --managed-python --no-config --clear --python "$PYTHON_VERSION" "$CANDIDATE_VENV"
export UV_PROJECT_ENVIRONMENT="$CANDIDATE_VENV"
"$UV_BIN" sync --managed-python --no-config --frozen --python "$PYTHON_VERSION" --project "$WORKTREE" --link-mode clone
assert_no_workspace_mounts
verify_worktree_closure
cd "$WORKTREE"
"$UV_BIN" run --managed-python --no-config --frozen --no-sync --no-env-file --project "$WORKTREE" --python "$PYTHON_VERSION" "$GENERATOR_REL" \
    --workspace-root "$WORKSPACE_ROOT" \
    --metadata-root "$WORKTREE/src/robomme/env_metadata/train" \
    --max-seed-attempts 1 \
    --save-video \
    "${candidate_args[@]}"
assert_no_workspace_mounts
verify_worktree_closure

output_relative="${normalized_output_dir#"$GENERATED_BASE"/}"
contract_report="$WORKSPACE_ROOT/artifacts/reports/generated/$CANDIDATE_COMMIT/$output_relative/generation_contract.json"
validator_args=(
    --generated-dir "$normalized_output_dir"
    --metadata-root "$WORKTREE/src/robomme/env_metadata/train"
    --workspace-root "$WORKSPACE_ROOT"
    --report "$contract_report"
    --expected-episodes "$episodes"
)
for env_id in "${selected_envs[@]}"; do
    validator_args+=(--expected-env "$env_id")
done

cd "$WORKSPACE_ROOT"
assert_no_workspace_mounts
"$UV_BIN" run --managed-python --no-config --frozen --no-sync --no-env-file --project "$WORKTREE" --python "$PYTHON_VERSION" "$VALIDATOR" "${validator_args[@]}"
