# ============================================================================
# cs336_basics C++ 扩展
#
# 构建一个 pybind11 扩展模块 `cs336_basics._cs336_ops`，供 Python 侧
# train_bpe / tokenizer / serialization 调用。
#
# 编译器：clang++   链接器：lld（由 toolchain.cmake 统一施加）
# 标准库：不做强制，用 clang 的默认值，即系统 libstdc++
#
# 快速使用（从仓库根目录）：
#     bash cs336_basics/cpp/build.sh
# ============================================================================
cmake_minimum_required(VERSION 3.24)

project(cs336_ops LANGUAGES CXX)

# --------------------------------------------------------------------------
# C++ 标准
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # 用 -std=c++20 而不是 -std=gnu++20

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# --------------------------------------------------------------------------
# 找到项目 .venv 里的 Python
#
# pybind11 的 FindPython 会顺带查出 Python.h 路径和扩展名后缀
# （例如 .cpython-313-x86_64-linux-gnu.so），不要自己拼后缀。
# 需要跨 Python 构建时：cmake -DPython_EXECUTABLE=/path/to/python ...
# --------------------------------------------------------------------------
if(NOT Python_EXECUTABLE)
    set(_candidates
        "${CMAKE_CURRENT_LIST_DIR}/../../../.venv/bin/python"   # 仓库根的 .venv
        "${CMAKE_CURRENT_LIST_DIR}/../../.venv/bin/python"
    )
    foreach(_p IN LISTS _candidates)
        if(EXISTS "${_p}")
            set(Python_EXECUTABLE "${_p}")
            break()
        endif()
    endforeach()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# --------------------------------------------------------------------------
# 找到 pybind11
#
# 首选：Python 环境里 pip/uv 装的 pybind11（会带上正确的 headers）
# 回退：CMAKE_PREFIX_PATH 里已有的 pybind11Config.cmake
# --------------------------------------------------------------------------
if(NOT pybind11_DIR AND Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _pybind11_probe
    )
    if(_pybind11_probe EQUAL 0 AND _pybind11_cmakedir)
        set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 cmake dir")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 ${pybind11_VERSION}  (${pybind11_DIR})")

# --------------------------------------------------------------------------
# 源文件
#
# 加新的 .cpp 就直接往这个列表里加；每个文件负责自己的实现。
# 当前 cs336_ops.cpp 里是 train_bpe（pre-token 计数 + BPE 合并）。
# --------------------------------------------------------------------------
set(CS336_OPS_SOURCES
    src/cs336_ops.cpp
)

pybind11_add_module(_cs336_ops ${CS336_OPS_SOURCES})

# --------------------------------------------------------------------------
# 编译选项
# --------------------------------------------------------------------------
target_compile_options(_cs336_ops PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-O0 -g3>
    # 需要跑 perf / 火焰图时打开：保留帧指针
    # -fno-omit-frame-pointer
)

target_include_directories(_cs336_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 位置无关代码（扩展模块必须是 PIC）
set_target_properties(_cs336_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------------
# 输出位置
#
# 直接落到包目录里，这样 `import cs336_basics._cs336_ops` 不用改 sys.path。
# 最终文件形如 cs336_basics/_cs336_ops.cpython-313-x86_64-linux-gnu.so，
# .gitignore 里的 *.so 规则会把它挡在版本控制外。
# --------------------------------------------------------------------------
set_target_properties(_cs336_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)

# --------------------------------------------------------------------------
# 自检信息：configure 结束后打印一遍，方便确认工具链真的生效了
# --------------------------------------------------------------------------
message(STATUS "-----------------------------------------------------------")
message(STATUS "  Python      : ${Python_EXECUTABLE}  (${Python_VERSION})")
message(STATUS "  C++ 编译器   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ 标准     : c++${CMAKE_CXX_STANDARD}")
message(STATUS "  链接器       : ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "  构建类型     : ${CMAKE_BUILD_TYPE}")
message(STATUS "-----------------------------------------------------------")
