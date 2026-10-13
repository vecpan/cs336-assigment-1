# ============================================================================
# toolchain.cmake —— 工具链文件（toolchain file）
#
# 用法：在 project() 之前用 -DCMAKE_TOOLCHAIN_FILE=... 传入。
# 只要这个文件在 project() 之前被 include，CMake 就会在这里做编译器探测，
# 所以下面的 *-FLAGS 变量能被自动加到所有 target 上。
#
# 这个文件负责两件事：
#   1. 强制用 clang++ 而不是 gcc
#   2. 强制用 lld 做链接（而不是 GNU ld / gold）
#
# 标准库不做任何强制，用 clang 在这台机器上的默认值（libstdc++）。
# 如果以后要改用 libc++，必须给编译和链接两个阶段都加 -stdlib=libc++，
# 只加一边会出现“头文件是 libc++、符号是 libstdc++”的混合状态。
# ============================================================================

# ---------------------------------------------------------------------------
# 1) 编译器：clang
# ---------------------------------------------------------------------------
set(CMAKE_C_COMPILER   "clang"   CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "" FORCE)

# ---------------------------------------------------------------------------
# 共享库 / 可执行文件的链接器统一设成 lld
#
# 注意：这里不要用 CMAKE_EXE_LINKER_FLAGS 的 "-fuse-ld=lld" 硬写，
# 因为 Clang 23 起 -fuse-ld=lld 走的是 linker driver 机制，两种写法都支持，
# 但显式声明 -fuse-ld=lld 语义最清楚，也和 --ld-path 不冲突。
# ---------------------------------------------------------------------------
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT    "-fuse-ld=lld" CACHE STRING "" FORCE)
