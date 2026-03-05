#coding:utf-8
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Tuple, Dict, Any

class QwenDataProcess:
    def __init__(self, custom_args, tokenizer, is_train):
        self.max_length = custom_args.max_length
        self.max_source_length = custom_args.max_source_length
        self.max_target_length = custom_args.max_target_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.instruction_column = custom_args.instruction_column ##system
        self.input_column = custom_args.input_column ##user
        self.output_column = custom_args.output_column ##answer
        self.training_mode = custom_args.training_mode #training_mode
        # 每个 item 的 codeword 数，默认 3，可在 custom_args.ldpo_m 中覆盖
        self.ldpo_m = getattr(custom_args, "ldpo_m", 3)

    def filter_fn(self, example: Dict[str, Any]) -> bool:
        """用于过滤无效样本：input_column 不为 None 且非空字符串"""
        input_val = example.get(self.input_column)
        answer_val = example.get(self.output_column)
        return isinstance(input_val, str) and len(input_val.strip()) > 0  \
            and isinstance(answer_val, str) and len(answer_val.strip()) > 0

    def __call__(self, example):
        # IAP+LDPO 模式：保持与原先相同的对话模板，仅重写 response 段为多 item codeword，
        # 并补充 ldpo_* 字段。
        if self.training_mode == "iap_ldpo":
            IGNORE_TOKEN_ID = LabelSmoother.ignore_index
            IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
            im_start = self.tokenizer(IM_START).input_ids
            im_end = self.tokenizer(IM_END).input_ids
            nl_tokens = self.tokenizer(NL).input_ids  ## 换行符
            _system = self.tokenizer("system").input_ids + nl_tokens
            _user = self.tokenizer("user").input_ids + nl_tokens
            _assistant = self.tokenizer("assistant").input_ids + nl_tokens
            default_instruction = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            )

            system_text = example.get(self.instruction_column)
            if not system_text:
                system_text = default_instruction
            user_text = example.get(self.input_column, "")
            answer_text = example.get(self.output_column, "")

            # 1) system 段（与 _encode_data 一致）
            input_ids = im_start + _system + self.tokenizer(system_text).input_ids + im_end + nl_tokens
            labels = (
                [IGNORE_TOKEN_ID] * len(im_start)
                + [IGNORE_TOKEN_ID] * (len(input_ids) - 3)
                + [IGNORE_TOKEN_ID] * len(im_end)
                + [IGNORE_TOKEN_ID] * len(nl_tokens)
            )

            # 2) user 段（与 _encode_data 一致）
            prompt_ids = self._tokenize_text(user_text, max_length=self.max_source_length)
            user_segment = im_start + _user + prompt_ids + im_end + nl_tokens + im_start + _assistant
            user_labels = [IGNORE_TOKEN_ID] * len(user_segment)

            input_ids += user_segment
            labels += user_labels

            # 3) response 段：answer 中是逗号分隔的多个商品，改造成多 item codewords
            #    格式： [item1_tokens][item2_tokens]...[itemN_tokens] IM_END NL
            ldpo_item_index: List[int] = [-1] * len(input_ids)

            # 解析 answer 中的多个商品
            raw_items = [s.strip() for s in answer_text.split(",") if s.strip()]
            m = int(self.ldpo_m) if self.ldpo_m and self.ldpo_m > 0 else 3

            ldpo_item_groups: List[int] = []
            item_id = 0

            for i, item_text in enumerate(raw_items):
                item_token_ids = self._tokenize_text(item_text, max_length=self.max_target_length)
                # 保证每个 item 最多 m 个 codeword；不足 m 的直接用实际长度
                if len(item_token_ids) > m:
                    item_token_ids = item_token_ids[:m]
                if len(item_token_ids) == 0:
                    continue

                # item 的 codeword token：参与 NTP，属于同一 item_id
                start_pos = len(input_ids)
                input_ids += item_token_ids
                labels += item_token_ids
                ldpo_item_index += [item_id] * len(item_token_ids)

                # 每个 item 的分组：answer 已按优先级从高到低排序，
                # 默认将前三个位置映射为 3,2,1（不足三个时按 3,2 或 3 处理）
                group = 3 - i
                if group < 1:
                    group = 1
                ldpo_item_groups.append(group)
                item_id += 1

            # 4) 末尾 IM_END + NL（与 _encode_data 一致，视为历史）
            tail_ids = im_end + nl_tokens
            input_ids += tail_ids
            labels += [IGNORE_TOKEN_ID] * len(tail_ids)
            ldpo_item_index += [-1] * len(tail_ids)

            # 5) 全局截断到 max_length，并保持三个列表对齐
            if (
                self.max_length
                and self.max_length > 0
                and self.max_length < len(input_ids)
            ):
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                ldpo_item_index = ldpo_item_index[: self.max_length]

            # 6) 训练时在末尾拼接 eos（保持与原先逻辑一致），视为历史
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

        # 其它模式保持原先逻辑
        input = example[self.input_column]
        if self.output_column in example:
            output = example[self.output_column]
        instruction = None
        if self.instruction_column in example:
            instruction = example[self.instruction_column]
        if self.training_mode == 'pretrain':
            data_list = self._encode_data_for_pretrain(input)
        else:
            data_list = self._encode_data(input, output if self.is_train else None, instruction)

        input_ids, labels = [], []
        for _, input_id, label in data_list:
            input_ids += input_id
            labels += label

        if self.max_length and self.max_length > 0 and self.max_length < len(input_ids):
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        if self.is_train:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _tokenize_text(self, text: str, max_length: int = None) -> List[int]:
        if text is None or len(text) == 0:
            text = " "
        ids = self.tokenizer(text).input_ids
        if max_length > 0 and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    # from git: https://github.com/QwenLM/Qwen-7B/blob/main/finetune.py#L123
    def _encode_data(self, input: str, output: str,
                     instruction: str) -> List[Tuple[str, List[int], List[int]]]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START).input_ids
        im_end = self.tokenizer(IM_END).input_ids
        nl_tokens = self.tokenizer(NL).input_ids ## 换行符
        _system = self.tokenizer('system').input_ids + nl_tokens
        _user = self.tokenizer('user').input_ids + nl_tokens
        _assistant = self.tokenizer('assistant').input_ids + nl_tokens
        default_instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        data_list = []
        # instruct,
        if not instruction:
            instruction = default_instruction
        input_ids = im_start + _system + self.tokenizer(instruction).input_ids + im_end + nl_tokens
        labels = [IGNORE_TOKEN_ID] * len(im_start) + [IGNORE_TOKEN_ID] * (len(input_ids) - 3) + [IGNORE_TOKEN_ID] * len(im_end) + [IGNORE_TOKEN_ID] * len(nl_tokens)
        data_list.append(("instruct", input_ids, labels)) ## system

        assert len(input_ids) == len(labels)
        # conversations
        history = []
        history.append([input, output])
        for turn_prompt, turn_response in history:
            # prompt IGNORE_TOKEN_ID
            prompt_ids = self._tokenize_text(turn_prompt, max_length=self.max_source_length)
            input_ids = im_start + _user + prompt_ids + im_end + nl_tokens + im_start + _assistant
            labels = [IGNORE_TOKEN_ID]*len(im_start) + [IGNORE_TOKEN_ID] * len(_user + prompt_ids) + [IGNORE_TOKEN_ID]*(len(im_end) + len(nl_tokens) + len(im_start)) + [
                IGNORE_TOKEN_ID] * len(_assistant)
            assert len(input_ids) == len(labels)
            data_list.append(("prompt", input_ids, labels))

            # response LABEL
            if turn_response:
                response_ids = self._tokenize_text(turn_response, max_length=self.max_target_length)
                input_ids = response_ids + im_end + nl_tokens
                labels = response_ids + [IGNORE_TOKEN_ID] * (len(im_end) + len(nl_tokens))
                assert len(input_ids) == len(labels)
                data_list.append(("response", input_ids, labels))
        return data_list


    def _encode_data_for_pretrain(self, input: str):
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        data_list = []

        # response LABEL
        response_ids = self._tokenize_text(input, max_length=self.max_target_length)
        input_ids = response_ids
        labels = response_ids
        assert len(input_ids) == len(labels)
        data_list.append(("corpus", input_ids, labels))
        return data_list