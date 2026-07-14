#!/usr/bin/env bash
set -euo pipefail

readonly CANDIDATE_COMMIT="a3842d1b77bc79e2f70cefcbab136207e7067065"
readonly PYTHON_VERSION="3.11.14"
readonly PATCH_SHA256="0336aa404ce805a160986857763ad89dbe72990d3afe662084a0d08d9c20c366"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
WORKTREE="$WORKSPACE_ROOT/artifacts/recovery/worktrees/${CANDIDATE_COMMIT}-recovered-v2"
PATCH_FILE="$WORKSPACE_ROOT/recovery/$CANDIDATE_COMMIT/generator-repo-local.patch"
GENERATOR_REL="scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py"
VALIDATOR="$WORKSPACE_ROOT/scripts/validate_generated_dataset_contract.py"
GENERATED_BASE="$(realpath -m -- "$WORKSPACE_ROOT/artifacts/generated/$CANDIDATE_COMMIT")"

usage() {
    cat <<'EOF'
用法：
  scripts/run_recovered_dataset_generator.sh --output-dir <候选专用的新目录> [生成器参数]

示例：
  scripts/run_recovered_dataset_generator.sh \
    --output-dir artifacts/generated/a3842d1/official-train-episodes-0-8 \
    --episodes 9 --max-workers 9 --gpus 0,1 --save-video

运行器固定以下参数，调用方不得覆盖：
  --workspace-root、--metadata-root、--max-seed-attempts、记录模式
EOF
}

if ! command -v uv >/dev/null 2>&1; then
    echo "错误：当前 workspace 未找到 uv。" >&2
    exit 2
fi
if [[ ! -f "$WORKSPACE_ROOT/pyproject.toml" || ! -f "$WORKSPACE_ROOT/uv.lock" ]]; then
    echo "错误：workspace 缺少 pyproject.toml 或 uv.lock。" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

saw_output_dir=false
help_requested=false
generator_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace-root|--workspace-root=*|--metadata-root|--metadata-root=*|--max-seed-attempts|--max-seed-attempts=*)
            echo "错误：参数 $1 由恢复运行器固定，不能覆盖。" >&2
            exit 2
            ;;
        --no-save-video)
            echo "错误：历史入口的 --no-save-video 同时关闭 timestep 记录，不能用于恢复官方 dataset。" >&2
            exit 2
            ;;
        --save-video)
            shift
            continue
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "错误：--output-dir 缺少路径。" >&2
                exit 2
            fi
            raw_output_dir="$2"
            shift 2
            ;;
        --output-dir=*)
            raw_output_dir="${1#--output-dir=}"
            shift
            ;;
        --help|-h)
            help_requested=true
            generator_args+=("$1")
            shift
            continue
            ;;
        *)
            generator_args+=("$1")
            shift
            continue
            ;;
    esac

    if [[ -z "$raw_output_dir" ]]; then
        echo "错误：--output-dir 不能为空。" >&2
        exit 2
    fi
    if [[ "$raw_output_dir" = /* ]]; then
        normalized_output_dir="$(realpath -m -- "$raw_output_dir")"
    else
        normalized_output_dir="$(realpath -m -- "$WORKSPACE_ROOT/$raw_output_dir")"
    fi
    generator_args+=(--output-dir "$normalized_output_dir")
    if [[ "$saw_output_dir" == true ]]; then
        echo "错误：--output-dir 只能传入一次。" >&2
        exit 2
    else
        saw_output_dir=true
    fi
done
if [[ "$saw_output_dir" != true ]]; then
    echo "错误：必须显式传入 --output-dir。" >&2
    exit 2
fi
case "$normalized_output_dir" in
    "$GENERATED_BASE"/*)
        ;;
    *)
        echo "错误：--output-dir 必须是 $GENERATED_BASE 的独立子目录；得到 $normalized_output_dir。" >&2
        exit 2
        ;;
esac
if [[ -e "$normalized_output_dir" && ! -d "$normalized_output_dir" ]]; then
    echo "错误：--output-dir 已存在且不是目录：$normalized_output_dir" >&2
    exit 2
fi
if [[ -d "$normalized_output_dir" ]] && [[ -n "$(find "$normalized_output_dir" -mindepth 1 -print -quit)" ]]; then
    echo "错误：--output-dir 必须不存在或为空，拒绝复用旧生成数据：$normalized_output_dir" >&2
    exit 2
fi

export UV_CACHE_DIR="$WORKSPACE_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE_ROOT/.cache/uv-python"
mkdir -p "$(dirname "$WORKTREE")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if [[ ! -e "$WORKTREE/.git" ]]; then
    git -C "$WORKSPACE_ROOT" worktree add --detach "$WORKTREE" "$CANDIDATE_COMMIT"
fi

actual_commit="$(git -C "$WORKTREE" rev-parse HEAD)"
if [[ "$actual_commit" != "$CANDIDATE_COMMIT" ]]; then
    echo "错误：恢复 worktree HEAD 为 $actual_commit，预期 $CANDIDATE_COMMIT。" >&2
    exit 2
fi

actual_patch_sha256="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$PATCH_SHA256" ]]; then
    echo "错误：恢复补丁 SHA-256 为 $actual_patch_sha256，预期 $PATCH_SHA256。" >&2
    exit 2
fi

if git -C "$WORKTREE" apply --unidiff-zero --reverse --check "$PATCH_FILE" 2>/dev/null; then
    :
elif git -C "$WORKTREE" apply --unidiff-zero --check "$PATCH_FILE"; then
    git -C "$WORKTREE" apply --unidiff-zero "$PATCH_FILE"
else
    echo "错误：恢复补丁既无法应用，也不是已应用状态；拒绝覆盖现有修改。" >&2
    exit 2
fi

if ! git -C "$WORKTREE" diff --cached --quiet; then
    echo "错误：恢复 worktree 存在 staged 修改。" >&2
    exit 2
fi
mapfile -t changed_files < <(git -C "$WORKTREE" diff --name-only HEAD)
for changed_file in "${changed_files[@]}"; do
    if [[ "$changed_file" != "$GENERATOR_REL" ]]; then
        echo "错误：恢复 worktree 存在补丁范围外修改：$changed_file" >&2
        exit 2
    fi
done
mapfile -t untracked_files < <(git -C "$WORKTREE" ls-files --others --exclude-standard)
if [[ ${#untracked_files[@]} -gt 0 ]]; then
    printf '错误：恢复 worktree 存在未跟踪文件：%s\n' "${untracked_files[@]}" >&2
    exit 2
fi
if ! cmp -s "$PATCH_FILE" <(
    git -C "$WORKTREE" diff --no-color --full-index --unified=0 HEAD -- "$GENERATOR_REL"
); then
    echo "错误：恢复 worktree 的完整生成器 diff 与固化补丁不一致。" >&2
    exit 2
fi
git -C "$WORKTREE" diff --check

if [[ ! -f "$WORKTREE/pyproject.toml" || ! -f "$WORKTREE/uv.lock" ]]; then
    echo "错误：候选 worktree 缺少 pyproject.toml 或 uv.lock。" >&2
    exit 2
fi
if [[ ! -f "$VALIDATOR" ]]; then
    echo "错误：缺少生成后契约验证器：$VALIDATOR" >&2
    exit 2
fi

uv python install "$PYTHON_VERSION"
cd "$WORKTREE"
uv run --frozen --python "$PYTHON_VERSION" "$GENERATOR_REL" \
    --workspace-root "$WORKSPACE_ROOT" \
    --metadata-root "$WORKTREE/src/robomme/env_metadata/train" \
    --max-seed-attempts 1 \
    --save-video \
    "${generator_args[@]}"

if [[ "$help_requested" == true ]]; then
    exit 0
fi

output_relative="${normalized_output_dir#"$GENERATED_BASE"/}"
contract_report="$WORKSPACE_ROOT/artifacts/reports/generated/$CANDIDATE_COMMIT/$output_relative/generation_contract.json"
cd "$WORKSPACE_ROOT"
uv run --frozen "$VALIDATOR" \
    --generated-dir "$normalized_output_dir" \
    --metadata-root "$WORKTREE/src/robomme/env_metadata/train" \
    --workspace-root "$WORKSPACE_ROOT" \
    --report "$contract_report"
