#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
 
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>

#include <string>
#include "BS_thread_pool.hpp"

namespace py = pybind11;

namespace cs336_ops {

// 把 [start, end) 里的 pre-token 计数合并进 counts。
void count_range(
    const std::vector<std::string>& pretokens,
    std::size_t start,
    std::size_t end,
    std::unordered_map<std::string, std::uint64_t>& counts) {
    for (std::size_t i = start; i < end; ++i) {
        ++counts[pretokens[i]];
    }
}

// ----------------------------------------------------------------------------
// train_bpe：从 pre-token 列表训练 BPE，返回 (vocab, merges)。
//
//   vocab  : dict[int, bytes]，token id -> token bytes。前 256 个是单字节 token，
//            之后依次是 special tokens 和训练中合并出来的新 token。
//   merges : list[tuple[bytes, bytes]]，按创建顺序排列的合并对。
//
// 计数阶段释放 GIL，并按线程池的线程数切块并行：每个 block 产出自己的局部
// map，worker 之间零共享；主线程最后串行合并。没有创建线程池时退化为串行，
// 保证结果与并行一致。
// ----------------------------------------------------------------------------
py::tuple train_bpe(const std::vector<std::string>& pretokens, uint32_t vocab_size, const std::vector<std::string>& special_tokens) {

    std::unordered_map<std::string, std::uint64_t> counts;

    {
        // 先放 GIL
        py::gil_scoped_release release;

        auto thread_pool = std::make_unique<BS::thread_pool<>>();
        const std::size_t total = pretokens.size();

        if (!thread_pool || total == 0) {
            count_range(pretokens, 0, total, counts);
        } else {
            // worker 全程零共享：每个 block 返回自己的局部 map，
            // 主线程拿到 multi_future 后串行合并。
            auto futures = thread_pool->submit_blocks(
                std::size_t{0}, total,
                [&](std::size_t block_start, std::size_t block_end) {
                    std::unordered_map<std::string, std::uint64_t> local;
                    local.reserve(block_end - block_start);
                    count_range(pretokens, block_start, block_end, local);
                    return local;
                });

            for (auto& future : futures) {
                const auto local = future.get();
                for (const auto& [key, value] : local) {
                    counts[key] += value;
                }
            }
        }
        thread_pool.reset();
    }

    std::map<uint32_t, std::string> vocabulary;
    uint32_t token_id = 0U;
    for (uint32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
        vocabulary.emplace(token_id++, std::string(1, static_cast<char>(i)));
    }
    for (const auto& special_token : special_tokens) {
        vocabulary[token_id++] = special_token;
    }

    // merge 阶段：朴素实现——每个 step 扫描一次整个 words 列表，把加权频次最高
    // 的相邻 token pair 合并成一个新 token，直到 vocabulary 填满。
    // 频次相同时取字典序更大的 pair。
    struct Word {
        std::vector<uint32_t> tokens;
        std::uint64_t freq;
    };

    std::vector<Word> words;
    words.reserve(counts.size());
    for (const auto& [pretoken, freq] : counts) {
        Word word;
        word.tokens.reserve(pretoken.size());
        for (const unsigned char byte : pretoken) {
            word.tokens.push_back(static_cast<uint32_t>(byte));
        }
        word.freq = freq;
        words.push_back(std::move(word));
    }

    py::list merges;
    while (vocabulary.size() != vocab_size) {
        // 1) 扫描所有 word，统计相邻 pair 的加权频次。
        std::map<std::pair<uint32_t, uint32_t>, std::uint64_t> pair_counts;
        for (const auto& word : words) {
            for (std::size_t i = 1; i < word.tokens.size(); ++i) {
                pair_counts[{word.tokens[i - 1], word.tokens[i]}] += word.freq;
            }
        }
        if (pair_counts.empty()) {
            break;
        }


        // 2) 取频次最高者；并列时取字典序更大的 pair。
        auto best = pair_counts.begin();
        for (auto it = std::next(pair_counts.begin()); it != pair_counts.end(); ++it) {
            if (it->second < best->second) {
                continue;
            }
            if (it->second > best->second ) {
                best = it;
                continue;
            }

            // 并列时按两个 token 的 bytes 做字典序比较：it 是候选，best 是当前最优。
            const auto cand_left = vocabulary.at(it->first.first);
            const auto cand_right = vocabulary.at(it->first.second);
            const auto best_left = vocabulary.at(best->first.first);
            const auto best_right = vocabulary.at(best->first.second);
            if (cand_left > best_left || (cand_left == best_left && cand_right > best_right)) {
                best = it;
            }
        }

        const uint32_t left = best->first.first;
        const uint32_t right = best->first.second;
        auto left_token = vocabulary[left];
        auto right_token = vocabulary[right];

        // 3) 登记新 token：id 顺延，新 token 的 bytes 是两个待合并 token 的拼接。
        const uint32_t new_id = static_cast<uint32_t>(vocabulary.size());
        vocabulary.emplace(new_id, left_token + right_token);

        merges.append(py::make_tuple(py::bytes(left_token.data(), left_token.size()),
                         py::bytes(right_token.data(), right_token.size())));


        // 4) 重写每个 word：把不重叠的 (left, right) 替换为 new_id。
        for (auto& word : words) {
            std::vector<uint32_t> merged;
            merged.reserve(word.tokens.size());
            for (std::size_t i = 0; i < word.tokens.size();) {
                if (i + 1 < word.tokens.size() &&
                    word.tokens[i] == left && word.tokens[i + 1] == right) {
                    merged.push_back(new_id);
                    i += 2;
                } else {
                    merged.push_back(word.tokens[i]);
                    ++i;
                }
            }
            word.tokens.swap(merged);
        }
    }

    // 返回值：vocabulary 为 id -> bytes 的映射，merges 按创建顺序排列。
    py::dict result;
    for (const auto& [key, value] : vocabulary) {
        py::bytes token(value.data(), value.size());
        result[py::int_(key)] = token;
    }

    return py::make_tuple(result, merges);
}
}  // namespace cs336_ops

// ============================================================================
// 模块定义
//
// 第一个参数必须和 target 名 / 输出文件名一致，即 `_cs336_ops`，
// 否则 import 时会报 dynamic module does not define module export function。
// ============================================================================
PYBIND11_MODULE(_cs336_ops, m) {
    m.doc() = "CS336 assignment 1 的 C++ 加速模块";

    // ------------------------------------------------------------------------
    // 加新函数时：先在 namespace cs336_ops 里写好实现，再在这里用 m.def 挂出来。
    // py::arg(...) 声明的参数名要和 Python 侧的调用方式一致，关键字传参依赖它。
    // ------------------------------------------------------------------------
    m.def("train_bpe", &cs336_ops::train_bpe,
          py::arg("pretokens"),
          py::arg("vocab_size"),
          py::arg("special_tokens"),
         "从 pre-token 列表训练 BPE，返回 (vocab, merges)：\n"
         "vocab 为 dict[int, bytes]，merges 为按创建顺序排列的 list[tuple[bytes, bytes]]。");
}
