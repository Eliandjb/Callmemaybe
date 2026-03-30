import json
from itertools import chain
from typing import List

from llm_sdk import Small_LLM_Model
from models import Calling, Function


class Startword:

    def __init__(self):
        self.llm_model = Small_LLM_Model()

    def test(self, list_func: List[Function], list_callable: List[Calling]):
        list_input = []
        for x in list_func:
            list_input.append(x.name)
        token_list = []
        mask_func = list_input
        for token in list_input:
            token_list.append(list(self.llm_model.encode(token)[0]))
        tools_json = json.dumps(
            [f.model_dump() for f in list_func], indent=None
        )
        full_prompt = (
            "[SYSTEM_INSTRUCTION]\n"
            "You are a deterministic JSON generator for tool selection. "
            "Your goal is to transform the user query into a JSON array "
            "containing a single object.\n"
            "[CONSTRAINTS]\n"
            "1. Respond ONLY with a valid JSON array.\n"
            "2. No text before or after the JSON.\n"
            "3. Numeric values must be represented as floats (e.g., 2.0).\n"
            "4. Use the exact keys: \"prompt\", \"name\", \"parameters\".\n"
            "[AVAILABLE_TOOLS]\n"
            f"{tools_json}\n"
            "[OUTPUT_FORMAT_EXAMPLE]\n"
            "\n  {\n    \"prompt\": \"User query here\",\n"
            "    \"name\": \"function_name\",\n"
            "    \"parameters\": {\"key\": value}\n  }\n\n"
        )
        static_encoded = list(self.llm_model.encode(full_prompt)[0])
        encode_func_list = []
        quote_id = self.llm_model.encode('"')[0][-1]
        id_bracket_close = self.llm_model.encode("}")[0][-1]
        id_bracket_open = self.llm_model.encode("{")[0][-1]
        for name_func in mask_func:
            name_func_str = str(name_func)
            encode_func_list.append(
                self.llm_model.encode(name_func_str)[0]
            )
            encode_func_list.append(
                self.llm_model.encode(" " + name_func_str)[0]
            )
        encode_func_value = set(chain.from_iterable(encode_func_list))
        encode_func_value.add(quote_id)
        with open("function_calling_results.json", "a") as fd:
            fd.write("[\n")
            for idx, x in enumerate(list_callable):
                escaped_prompt = x.prompt.replace('"', "'")
                add_prompt = (
                    "[USER_QUERY]\n"
                    f"{x.prompt}\n"
                    "[JSON_RESPONSE]\n"
                    f"\n  {{\n    \"prompt\": \"{escaped_prompt}\",\n"
                    "    \"name\": \""
                )
                dynamic_encoded = list(self.llm_model.encode(add_prompt)[0])
                generate_prompt_add = static_encoded + dynamic_encoded
                bracket_stack = 1
                generated_text = ""
                masking_flag = True
                flag_first = True
                for _ in range(50):
                    logits = self.llm_model.get_logits_from_input_ids(
                        generate_prompt_add
                    )
                    try:
                        logits_list = logits[0]
                        _ = len(logits_list)
                    except (TypeError, IndexError):
                        logits_list = logits
                    if masking_flag and flag_first is True:
                        v_size = len(logits_list)
                        valid_tokens = [
                            tid for tid in encode_func_value if tid < v_size
                        ]
                        flag_first = False
                    if masking_flag:
                        if valid_tokens:
                            logits_max_index = max(
                                valid_tokens,
                                key=lambda i: logits_list[i]
                            )
                        else:
                            max_val = max(logits_list)
                            logits_max_index = logits_list.index(max_val)
                    else:
                        max_val = max(logits_list)
                        logits_max_index = logits_list.index(max_val)
                    generate_prompt_add.append(logits_max_index)
                    token_text = self.llm_model.decode(logits_max_index)
                    generated_text += token_text
                    if logits_max_index == id_bracket_close:
                        bracket_stack -= 1
                    if logits_max_index == id_bracket_open:
                        bracket_stack += 1
                    if masking_flag and logits_max_index == quote_id:
                        masking_flag = False
                        injection_str = ',\n    "parameters": {'
                        injection_tokens = list(
                            self.llm_model.encode(injection_str)[0]
                        )
                        generate_prompt_add.extend(injection_tokens)
                        generated_text += injection_str
                        bracket_stack += 1
                    if bracket_stack <= 0:
                        break
                prefix = (
                    f"  {{\n    \"prompt\": \"{escaped_prompt}\",\n"
                    f"    \"name\": \""
                )
                clean_text = generated_text.rstrip()
                if clean_text.endswith("}"):
                    clean_text = clean_text[:-1].rstrip() + "\n  }"
                clean_text = clean_text.replace('\\', '\\\\')
                fd.write(prefix + clean_text)
                if idx < len(list_callable) - 1:
                    fd.write(",\n")
                else:
                    fd.write("\n")
            fd.write("]\n")
