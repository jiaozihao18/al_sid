from typing import Any, Dict, List, Tuple

from transformers.trainer_pt_utils import LabelSmoother


class QwenDataProcess:
    """
    - training_mode="iap_ldpo"：构造 IAP+LDPO 所需的 ldpo_* 字段
    - 其它情况：走标准对话模板，仅产出 input_ids / labels
    """

    def __init__(self, custom_args, tokenizer, is_train: bool):
        self.max_length = custom_args.get("max_length")
        self.max_source_length = custom_args.get("max_source_length")
        self.max_target_length = custom_args.get("max_target_length")
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.instruction_column = custom_args.get("instruction_column", "system")
        self.input_column = custom_args.get("input_column", "user")
        self.output_column = custom_args.get("output_column", "answer")
        self.training_mode = custom_args.get("training_mode")
        self.ldpo_m = custom_args.get("ldpo_m", 3)

    def filter_fn(self, example: Dict[str, Any]) -> bool:
        input_val = example.get(self.input_column)
        answer_val = example.get(self.output_column)
        return isinstance(input_val, str) and len(input_val.strip()) > 0 and isinstance(answer_val, str) and len(
            answer_val.strip()
        ) > 0

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # IAP+LDPO 模式
        if self.training_mode == "iap_ldpo":
            IGNORE_TOKEN_ID = LabelSmoother.ignore_index
            IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
            im_start = self.tokenizer(IM_START).input_ids
            im_end = self.tokenizer(IM_END).input_ids
            nl_tokens = self.tokenizer(NL).input_ids
            _system = self.tokenizer("system").input_ids + nl_tokens
            _user = self.tokenizer("user").input_ids + nl_tokens
            _assistant = self.tokenizer("assistant").input_ids + nl_tokens
            default_instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

            system_text = example.get(self.instruction_column) or default_instruction
            user_text = example.get(self.input_column, "")
            answer_text = example.get(self.output_column, "")

            # system 段
            input_ids = im_start + _system + self.tokenizer(system_text).input_ids + im_end + nl_tokens
            labels = (
                [IGNORE_TOKEN_ID] * len(im_start)
                + [IGNORE_TOKEN_ID] * (len(input_ids) - 3)
                + [IGNORE_TOKEN_ID] * len(im_end)
                + [IGNORE_TOKEN_ID] * len(nl_tokens)
            )

            # user 段
            prompt_ids = self._tokenize_text(user_text, max_length=self.max_source_length)
            user_segment = im_start + _user + prompt_ids + im_end + nl_tokens + im_start + _assistant
            user_labels = [IGNORE_TOKEN_ID] * len(user_segment)
            input_ids += user_segment
            labels += user_labels

            # response 段：多 item codewords
            ldpo_item_index: List[int] = [-1] * len(input_ids)

            raw_items = [s.strip() for s in answer_text.split(",") if s.strip()]
            m = int(self.ldpo_m) if self.ldpo_m and self.ldpo_m > 0 else 3

            ldpo_item_groups: List[int] = []
            item_id = 0

            for i, item_text in enumerate(raw_items):
                item_token_ids = self._tokenize_text(item_text, max_length=self.max_target_length)
                if len(item_token_ids) > m:
                    item_token_ids = item_token_ids[:m]
                if not item_token_ids:
                    continue

                input_ids += item_token_ids
                labels += item_token_ids
                ldpo_item_index += [item_id] * len(item_token_ids)

                group = 3 - i
                if group < 1:
                    group = 1
                ldpo_item_groups.append(group)
                item_id += 1

            tail_ids = im_end + nl_tokens
            input_ids += tail_ids
            labels += [IGNORE_TOKEN_ID] * len(tail_ids)
            ldpo_item_index += [-1] * len(tail_ids)

            if self.max_length and self.max_length > 0 and self.max_length < len(input_ids):
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                ldpo_item_index = ldpo_item_index[: self.max_length]

            if self.is_train:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    input_ids += [eos_id]
                    labels += [eos_id]
                    ldpo_item_index += [-1]

            ldpo_num_items = item_id

            return {
                "input_ids": input_ids,
                "labels": labels,
                "ldpo_item_index": ldpo_item_index,
                "ldpo_item_groups": ldpo_item_groups,
                "ldpo_num_items": ldpo_num_items,
                "ldpo_m": m,
            }

        # 非 LDPO：走对话模板，生成普通 input_ids / labels
        input_text = example.get(self.input_column, "")
        output_text = example.get(self.output_column, "")
        instruction = example.get(self.instruction_column)
        data_list = self._encode_data(input_text, output_text if self.is_train else None, instruction)

        input_ids: List[int] = []
        labels: List[int] = []
        for _, inp, lab in data_list:
            input_ids += inp
            labels += lab

        if self.max_length and self.max_length > 0 and self.max_length < len(input_ids):
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        if self.is_train:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_text(self, text: str, max_length: int = None) -> List[int]:
        if text is None or len(text) == 0:
            text = " "
        ids = self.tokenizer(text).input_ids
        if max_length and max_length > 0 and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def _encode_data(
        self,
        input_text: str,
        output_text: str,
        instruction: str,
    ) -> List[Tuple[str, List[int], List[int]]]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START).input_ids
        im_end = self.tokenizer(IM_END).input_ids
        nl_tokens = self.tokenizer(NL).input_ids
        _system = self.tokenizer("system").input_ids + nl_tokens
        _user = self.tokenizer("user").input_ids + nl_tokens
        _assistant = self.tokenizer("assistant").input_ids + nl_tokens
        default_instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        data_list: List[Tuple[str, List[int], List[int]]] = []

        if not instruction:
            instruction = default_instruction
        input_ids = im_start + _system + self.tokenizer(instruction).input_ids + im_end + nl_tokens
        labels = (
            [IGNORE_TOKEN_ID] * len(im_start)
            + [IGNORE_TOKEN_ID] * (len(input_ids) - 3)
            + [IGNORE_TOKEN_ID] * len(im_end)
            + [IGNORE_TOKEN_ID] * len(nl_tokens)
        )
        data_list.append(("instruct", input_ids, labels))

        history = [[input_text, output_text]]
        for turn_prompt, turn_response in history:
            prompt_ids = self._tokenize_text(turn_prompt, max_length=self.max_source_length)
            input_ids = im_start + _user + prompt_ids + im_end + nl_tokens + im_start + _assistant
            labels = (
                [IGNORE_TOKEN_ID] * len(im_start)
                + [IGNORE_TOKEN_ID] * len(_user + prompt_ids)
                + [IGNORE_TOKEN_ID] * (len(im_end) + len(nl_tokens) + len(im_start))
                + [IGNORE_TOKEN_ID] * len(_assistant)
            )
            data_list.append(("prompt", input_ids, labels))

            if turn_response:
                response_ids = self._tokenize_text(turn_response, max_length=self.max_target_length)
                input_ids = response_ids + im_end + nl_tokens
                labels = response_ids + [IGNORE_TOKEN_ID] * (len(im_end) + len(nl_tokens))
                data_list.append(("response", input_ids, labels))

        return data_list

