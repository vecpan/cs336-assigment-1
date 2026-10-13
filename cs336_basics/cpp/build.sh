#!/usr/bin/env bash

# ============================================================================
# build.sh —— 配置 + 编译 cs336_basics._cs336_ops
#
#     bash cs336_basics/cpp/build.sh              # Release（默认）
#     bash cs336_basics/cpp/build.sh Debug        # Debug，带 -g3 -O0
#     bash cs336_basics/cpp/build.sh clean        # 删掉 build 目录
#     bash cs336_basics/cpp/build.sh reconfigure  # 删掉 build 目录后重配
#
# 环境变量：
#     PYTHON=/path/to/python   指定用哪个解释器构建（默认仓库根的 .venv）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

# ---------------------------------------------------------------------------
# 干净重建
# ---------------------------------------------------------------------------
if [[ "${BUILD_TYPE}" == "clean" ]]; then
    rm -rf "${BUILD_DIR}"
    echo "已删除 ${BUILD_DIR}"
    exit 0
fi

if [[ "${BUILD_TYPE}" == "reconfigure" ]]; then
    rm -rf "${BUILD_DIR}"
    BUILD_TYPE="Release"
fi

# ---------------------------------------------------------------------------
# 选 Python：优先 $PYTHON，其次 .venv，最后系统 python3
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

# ---------------------------------------------------------------------------
# 前置检查：给出能直接照做的报错，而不是 CMake 的堆栈
# ---------------------------------------------------------------------------
command -v clang++ >/dev/null || { echo "错误：找不到 clang++" >&2; exit 1; }
command -v ld.lld  >/dev/null || { echo "错误：找不到 ld.lld（装 lld 包）" >&2; exit 1; }

if ! "${PYTHON_BIN}" -c "import pybind11" >/dev/null 2>&1; then
    echo "错误：${PYTHON_BIN} 里没有 pybind11" >&2
    echo "      装一个：uv pip install --python ${PYTHON_BIN} pybind11" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# configure + build
# ---------------------------------------------------------------------------
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain.cmake" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DPython_EXECUTABLE="${PYTHON_BIN}"

cmake --build "${BUILD_DIR}" --parallel
