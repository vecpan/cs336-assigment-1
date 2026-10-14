from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 pre-tokenization pattern: merges are only applied inside a pre-token, never
# across a whitespace/punctuation boundary.
GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BpeTokenizer:
    """Byte-level BPE tokenizer built from a vocabulary and an ordered merge list."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # `vocab` maps id -> bytes; keep both directions so that encode/decode never have
        # to scan the vocabulary backwards.
        self.id_to_token: dict[int, bytes] = vocab
        self.token_to_id: dict[bytes, int] = {token: token_id for token_id, token in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        # Special tokens are split off first, so that they stay a single piece and are
        # never touched by the merge loop. All remaining text is pre-tokenized below.
        if self.special_tokens:
            # Fail early: a special token without a vocabulary entry could not be mapped
            # back to an id once it has been split out.
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in self.token_to_id:
                    raise ValueError(f"special token {special_token!r} not found in vocab")

            # Longest special token first, so that "<|endoftext|><|endoftext|>" wins over
            # "<|endoftext|>" when both are configured.
            special_pattern = (
                "(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            # re.split with a capturing group keeps the special tokens in the result.
            segments = re.split(special_pattern, text)
        else:
            segments = [text]

        ids: list[int] = []
        for segment in segments:
            if segment == "":
                continue
            segment_bytes = segment.encode("utf-8")
            if self.special_tokens and segment_bytes in self.token_to_id:
                # A captured segment that is itself in the vocabulary is a special token:
                # it maps to its id directly, without pre-tokenization or merging.
                ids.append(self.token_to_id[segment_bytes])
            else:
                for pre_token in re.findall(GPT2_PRETOKEN_PATTERN, segment):
                    ids.extend(self._encode_pre_token(pre_token))

        return ids

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        """Apply byte-level BPE to a single pre-token and return its token ids."""
        # Work on bytes, not str: merges happen at the byte level and must not split a
        # character in the middle.
        data = pre_token.encode()
        tokens = [bytes([b]) for b in data]
        for merge in self.merges:
            merged_tokens: list[bytes] = []
            # just_merged means the previous position was consumed by the merge applied
            # there, so the token at the current position has to be skipped.
            just_merged = False
            for i in range(len(tokens) - 1):
                if just_merged:
                    just_merged = False
                    continue
                if merge[0] == tokens[i] and merge[1] == tokens[i + 1]:
                    merged_tokens.append(merge[0] + merge[1])
                    just_merged = True
                else:
                    merged_tokens.append(tokens[i])
            # The last token is still missing only when the last pair was not merged.
            if not just_merged and len(tokens) >= 1:
                merged_tokens.append(tokens[-1])
            tokens = merged_tokens

        ids: list[int] = []
        for token in tokens:
            if token not in self.token_to_id:
                raise ValueError(f"token {token!r} not found in vocab")
            ids.append(self.token_to_id[token])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        data = b""
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"token id {token_id} not found in original vocab")
            data += self.id_to_token[token_id]
        return data.decode("utf-8", errors="replace")

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """Stream ids: encode one string at a time instead of holding every id in memory."""
        for text in texts:
            yield from self.encode(text)
