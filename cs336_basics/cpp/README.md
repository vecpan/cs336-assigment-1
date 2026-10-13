# cs336_basics 的 C++ 扩展（pybind11）

用 **clang++ / lld** 构建 pybind11 扩展模块 `cs336_basics._cs336_ops`，
给 Python 侧的 `train_bpe`、`tokenizer`、`serialization` 调用。
标准库沿用 clang 的默认值（系统 **libstdc++**），本仓库不强制 libc++。

当前只实现了 `train_bpe`（pre-token 计数 + 朴素 BPE 合并）；分词、序列化等
接口的挂载点在 `src/cs336_ops.cpp` 底部的 `PYBIND11_MODULE` 里。

---

## 1. 目录结构

```
cs336_basics/cpp/
├── CMakeLists.txt        # 构建目标定义、pybind11 接线、输出位置
├── toolchain.cmake       # 工具链文件：强制 clang++ 和 lld
├── build.sh              # 一条命令完成 configure + build
├── .gitignore            # 挡掉 build/ 和编译产物
├── README.md             # 本文件
├── include/
│   └── BS_thread_pool.hpp  # 第三方线程池，train_bpe 计数阶段用它并行
├── src/
│   └── cs336_ops.cpp     # 模块源码（train_bpe）
└── build/                # CMake 生成物（不提交）
```

编译产物落在**包目录**而非 `cpp/build/`：

```
cs336_basics/_cs336_ops.cpython-313-x86_64-linux-gnu.so
```

这样 `import cs336_basics._cs336_ops` 不需要动 `sys.path`，
pytest 从仓库根目录跑的时候也能直接找到。

---

## 2. 构建

```bash
# 从仓库根目录
bash cs336_basics/cpp/build.sh              # Release（默认）
bash cs336_basics/cpp/build.sh Debug        # -O0 -g3
bash cs336_basics/cpp/build.sh clean        # 删掉 build/
bash cs336_basics/cpp/build.sh reconfigure  # 干净重配

# 换解释器构建
PYTHON=/usr/bin/python3 bash cs336_basics/cpp/build.sh
```

`build.sh` 会先做前置检查（`clang++`、`ld.lld`、`pybind11` 是否都在），
缺什么直接告诉你装什么，然后再 configure + build。
编译完可以按第 3 节里的命令手动 import 一次，确认产物能载入。

### 依赖：pybind11

当前环境用 uv 装在项目 `.venv` 里（版本 3.1.0）：

```bash
uv pip install --python .venv/bin/python pybind11
# 或者加进项目依赖：uv add --dev pybind11
```

`CMakeLists.txt` 会执行 `python -c "import pybind11; print(pybind11.get_cmake_dir())"`
自动定位，所以不需要手动传 `pybind11_DIR`。

### 手动用 CMake（不用 build.sh）

```bash
cmake -S cs336_basics/cpp -B cs336_basics/cpp/build -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=cs336_basics/cpp/toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=.venv/bin/python
cmake --build cs336_basics/cpp/build --parallel
```

---

## 3. 验证工具链真的生效了

光看“编译成功”不够——CMake 缓存里可能留着上一次配置的编译器。分两层验证：

**第一层，编译/链接命令**（证明用的是 clang++ 和 lld）：

```bash
cd cs336_basics/cpp/build
ninja -t commands _cs336_ops | grep -oE '\S*clang\+\+|-fuse-ld=lld' | sort -u
# -fuse-ld=lld
# /usr/bin/clang++
```

**第二层，产物**（确认 import 得进来，且运行时依赖符合预期）：

```bash
.venv/bin/python -c "from cs336_basics import _cs336_ops; print(dir(_cs336_ops))"
# [..., 'train_bpe']
ldd cs336_basics/_cs336_ops.cpython-313-x86_64-linux-gnu.so | grep -i "c++"
# libstdc++.so.6 => /usr/lib/libstdc++.so.6
readelf -d cs336_basics/_cs336_ops*.so | grep NEEDED
# 当前预期是 libstdc++.so.6；出现 libc++.so.1 说明链接阶段被别的配置改过
```

---

## 4. 三个关键设计点（为什么这么写）

### 4.1 为什么用工具链文件而不是在 CMakeLists 里 set

`*_FLAGS_INIT` 这类变量只在 `project()` 触发编译器探测**之前**有效。
工具链文件在 `project()` 之前被 include，所以这里设的值会被自动加进
所有 target 的编译和链接命令，不需要给每个 target 手写 `target_link_options`。

这也意味着**改完工具链文件必须重新 configure**（`build.sh reconfigure`），
否则 CMake 缓存里还是旧的编译器设置。

标准库这里刻意不做任何设置，用 clang 的默认值（本机是 libstdc++）。
如果以后要切到 libc++，需要**编译和链接两个阶段都加** `-stdlib=libc++`：
编译阶段决定头文件和 ABI 符号，链接阶段决定 `NEEDED` 里放哪个 `.so`，
只加一个就会出现“头文件是 libc++、符号是 libstdc++”的混合状态。

### 4.2 为什么输出到包目录

`LIBRARY_OUTPUT_DIRECTORY` 设为 `${CMAKE_CURRENT_LIST_DIR}/..`，
即 `cs336_basics/`。后缀（`.cpython-313-x86_64-linux-gnu.so`）由
`find_package(Python ... Development.Module)` 提供，**不要手写后缀**，
换 Python 版本就会失效。

### 4.3 模块边界的 ABI 注意事项

当前扩展和 CPython 一样用 libstdc++，进程里只有一个 C++ 标准库，
不存在双标准库的问题。但下面这条规则仍然成立：

> **模块边界上只暴露 pybind11 认识的基本类型**
> （`int`/`float`/`bool`/`str`/`bytes`/`list`/`dict`/`tuple`/`py::array_t`），
> 不要把 `std::string&`、`std::vector<T>&` 的引用、或自定义 C++ 类
> 跨过 Python 边界传来传去。

一是传内部 C++ 对象进来需要额外做 pybind11 绑定并自己管生命周期；
二是以后若改用 libc++，CPython（libstdc++）和扩展（libc++）的
`std::string`、`std::vector`、`std::exception` 内存布局和符号修饰都不兼容，
混用会得到**链接错误或难以定位的运行时崩溃**。

pybind11 默认是**按值转换**（`std::string` ↔ `str`），按这个用就没问题。
真要传大块数据（比如 token 数组）时，用 `py::array_t<uint16_t>` /
`py::buffer` 走缓冲区协议零拷贝，而不是 `std::vector` 来回转换。

---

## 5. 加你自己的实现

1. 新建 `src/xxx.cpp`（或直接写在 `cs336_ops.cpp` 里），把函数定义放在
   `namespace cs336_ops` 下。
2. 把新文件加进 `CMakeLists.txt` 的 `CS336_OPS_SOURCES`。
3. 在 `PYBIND11_MODULE(_cs336_ops, m)` 里用 `m.def(...)` 挂出来：

```cpp
m.def("train_bpe", &cs336_ops::train_bpe,
      py::arg("pretokens"), py::arg("vocab_size"), py::arg("special_tokens"));
```

4. `bash cs336_basics/cpp/build.sh` 重新构建。

**别改** `PYBIND11_MODULE` 的第一个参数（`_cs336_ops`）和 CMake target 名：
它俩必须一致，否则 import 时报
`dynamic module does not define module export function`。

### 性能相关的编译选项

优化级别由 `CMAKE_BUILD_TYPE` 控制。调性能时建议：

- 用 `Release`（`-O3`）测真实速度，不要用 `Debug` 的结论下判断。
- 要跑 perf / 火焰图时，在 `CMakeLists.txt` 里打开
  `-fno-omit-frame-pointer` 和 `RelWithDebInfo`（注释里已经留了位置）。
- pybind11 探测到 LTO 可用时会自动开 thin LTO，这是预期的。

---

## 6. 已知环境注意事项

- **换解释器会影响产物后缀**：`pyproject.toml` 的 `requires-python` 是
  `">=3.12,<3.14"`，当前 `.venv` 是 **3.13.16**（满足约束），
  `find_package(Python)` 报的也是 3.13，所以产物后缀是 `cpython-313`。
  如果用 `uv sync` 换了解释器，旧 `.so` 的文件名就对不上了，
  换完记得重新跑一次 `build.sh`。
- **标准库**：`build.sh` 产出的 `.so` 依赖系统 `libstdc++.so.6`
  （clang 在这台机器上的默认值），这是当前预期行为。系统里也装了
  libc++（`/usr/lib/libc++.so.1`），但 `toolchain.cmake` 没有启用它；
  真要切换时，记得编译和链接两个阶段都加 `-stdlib=libc++`。
- **libgcc_s**：产物 `NEEDED` 里有 `libgcc_s.so.1`，这是 clang 默认的
  unwinder 依赖，正常现象。
