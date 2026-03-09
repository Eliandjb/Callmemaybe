from llm_sdk import Small_LLM_Model
from typing import List
from models import Function, Calling
from pydantic import BaseModel
import json


class Startword():
    def __init__(self):
        self.llm_model = Small_LLM_Model()
    
    def test(self, list_func: List[Function], list_callable: List[Calling]):
        tools_json = json.dumps([f.model_dump() for f in list_func], indent=2)
        with open("function_calling_results.json", "a") as fd:
            for x in list_callable:
                full_prompt = (
                    "[SYSTEM_INSTRUCTION]\n"
                    "You are a highly accurate tool selection engine. "
                    "Your mission is to translate the user's query into a single JSON function call.\n"
                    "[CONSTRAINTS]\n"
                    "1. Respond ONLY in JSON.\n"
                    "2. Do not provide any explanation, introduction, or conclusion.\n"
                    "3. If no function matches, respond with: {\"error\": \"no_match\"}.\n"
                    "[AVAILABLE_TOOLS]\n" + tools_json + "\n"
                    "[OUTPUT_FORMAT]\n"
                    "You must follow this pattern: {\"name\": \"function_name\", \"arguments\": {\"key\": \"value\"}}\n"
                    "[USER_QUERY]\n"
                    f"{x.prompt}\n"
                    "[JSON_RESPONSE]\n"
                    "{"
                )
                print(full_prompt)
                input_id = []
                value = self.llm_model.encode(full_prompt)
                value2 = list(value[0])
                for _ in range(150):
                    logits = self.llm_model.get_logits_from_input_ids(value2)
                    logits_max_index = logits.index(max(logits))
                    value2.append(logits_max_index)
                    input_id.append(logits_max_index)
                fd.write(f"\nPrompt: {x.prompt}\nResponse: {{")
                for token in input_id:
                    text = self.llm_model.decode(token)
                    i = 0
                    while i + 1 < len(text):
                        if text[i] == "}" and text[i + 1] == "}":
                            text = text[0: i+1]
                    fd.write(f"{text}")
                fd.write("\n" + "-"*20)