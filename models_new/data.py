from typing import Any, Dict, List


IGNORE_INDEX = -100


class LDPODataProcess:
    """
    data preprocessing.

    Modes:
    - training_mode="iap_ldpo": produce LDPO fields
    - else: standard SFT conversation
    """

    def __init__(self, custom_args, tokenizer, is_train: bool):
        self.tokenizer = tokenizer
        self.is_train = is_train

        self.max_length = custom_args.get("max_length")
        self.max_source_length = custom_args.get("max_source_length")
        self.max_target_length = custom_args.get("max_target_length")

        self.instruction_column = custom_args.get("instruction_column", "system")
        self.input_column = custom_args.get("input_column", "user")
        self.output_column = custom_args.get("output_column", "answer")

        self.training_mode = custom_args.get("training_mode")
        self.ldpo_m = int(custom_args.get("ldpo_m", 3))

        self.ignore_token = IGNORE_INDEX

        # ---------- Pre-tokenized constants ----------
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"

        self.im_start = tokenizer(IM_START).input_ids
        self.im_end = tokenizer(IM_END).input_ids
        self.nl_tokens = tokenizer(NL).input_ids

        self.system_tokens = tokenizer("system").input_ids + self.nl_tokens
        self.user_tokens = tokenizer("user").input_ids + self.nl_tokens
        self.assistant_tokens = tokenizer("assistant").input_ids + self.nl_tokens

        self.default_instruction = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

        self.eos = tokenizer.eos_token_id

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:

        if self.training_mode == "iap_ldpo":
            return self._process_ldpo(example)

        return self._process_sft(example)

    # ------------------------------------------------
    # LDPO PROCESS
    # ------------------------------------------------

    def _process_ldpo(self, example):

        system_text = example.get(self.instruction_column) or self.default_instruction
        user_text = example.get(self.input_column, "")
        answer_text = example.get(self.output_column, "")

        input_ids: List[int] = []
        labels: List[int] = []
        ldpo_item_index: List[int] = []

        IGN = self.ignore_token

        # ---------- system ----------
        system_ids = self.tokenizer(system_text).input_ids

        seg = (
            self.im_start
            + self.system_tokens
            + system_ids
            + self.im_end
            + self.nl_tokens
        )

        input_ids.extend(seg)
        labels.extend([IGN] * len(seg))
        ldpo_item_index.extend([-1] * len(seg))

        # ---------- user ----------
        prompt_ids = self._tokenize(user_text, self.max_source_length)

        seg = (
            self.im_start
            + self.user_tokens
            + prompt_ids
            + self.im_end
            + self.nl_tokens
            + self.im_start
            + self.assistant_tokens
        )

        input_ids.extend(seg)
        labels.extend([IGN] * len(seg))
        ldpo_item_index.extend([-1] * len(seg))

        # ---------- response items ----------
        raw_items = [x.strip() for x in answer_text.split(",") if x.strip()]

        if raw_items:
            tokenized_items = self.tokenizer(
                raw_items,
                add_special_tokens=False
            ).input_ids
        else:
            tokenized_items = []

        ldpo_item_groups: List[int] = []
        item_id = 0

        for i, item_tokens in enumerate(tokenized_items):

            if self.ldpo_m > 0 and len(item_tokens) > self.ldpo_m:
                item_tokens = item_tokens[: self.ldpo_m]

            if not item_tokens:
                continue

            input_ids.extend(item_tokens)
            labels.extend(item_tokens)
            ldpo_item_index.extend([item_id] * len(item_tokens))

            group = max(1, 3 - i)
            ldpo_item_groups.append(group)

            item_id += 1

        # ---------- tail ----------
        tail = self.im_end + self.nl_tokens

        input_ids.extend(tail)
        labels.extend([IGN] * len(tail))
        ldpo_item_index.extend([-1] * len(tail))

        # ---------- truncate ----------
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            ldpo_item_index = ldpo_item_index[: self.max_length]

        # ---------- eos ----------
        if self.is_train and self.eos is not None:
            input_ids.append(self.eos)
            labels.append(self.eos)
            ldpo_item_index.append(-1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "ldpo_item_index": ldpo_item_index,
            "ldpo_item_groups": ldpo_item_groups,
            "ldpo_num_items": item_id,
            "ldpo_m": self.ldpo_m,
            "sample_type": 1,
        }

    # ------------------------------------------------
    # STANDARD SFT
    # ------------------------------------------------

    def _process_sft(self, example):

        instruction = example.get(self.instruction_column) or self.default_instruction
        user_text = example.get(self.input_column, "")
        answer_text = example.get(self.output_column, "")

        IGN = self.ignore_token

        input_ids: List[int] = []
        labels: List[int] = []

        # ---------- system ----------
        system_ids = self.tokenizer(instruction).input_ids

        seg = (
            self.im_start
            + self.system_tokens
            + system_ids
            + self.im_end
            + self.nl_tokens
        )

        input_ids.extend(seg)
        labels.extend([IGN] * len(seg))

        # ---------- user ----------
        prompt_ids = self._tokenize(user_text, self.max_source_length)

        seg = (
            self.im_start
            + self.user_tokens
            + prompt_ids
            + self.im_end
            + self.nl_tokens
            + self.im_start
            + self.assistant_tokens
        )

        input_ids.extend(seg)
        labels.extend([IGN] * len(seg))

        # ---------- assistant ----------
        if answer_text:

            resp_ids = self._tokenize(answer_text, self.max_target_length)

            seg = resp_ids + self.im_end + self.nl_tokens

            input_ids.extend(seg)

            labels.extend(
                resp_ids + [IGN] * (len(self.im_end) + len(self.nl_tokens))
            )

        # ---------- truncate ----------
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        # ---------- eos ----------
        if self.is_train and self.eos is not None:
            input_ids.append(self.eos)
            labels.append(self.eos)

        ldpo_item_index = [-1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "ldpo_item_index": ldpo_item_index,
            "ldpo_item_groups": [],
            "ldpo_num_items": 0,
            "ldpo_m": self.ldpo_m,
            "sample_type": 0,
        }

    # ------------------------------------------------

    def _tokenize(self, text: str, max_len=None):

        if not text:
            text = " "

        ids = self.tokenizer(text, add_special_tokens=False).input_ids

        if max_len and len(ids) > max_len:
            ids = ids[:max_len]

        return ids