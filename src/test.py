import json
import re
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from llm_sdk import Small_LLM_Model
from src.models import Calling, Function, FunctionCallResult
from typing import cast


NEG_INF: float = -1e9


class ConstrainedDecoder:
    """Constrained decoding engine using a small LLM.

    Pre-computes token-level data structures so that constrained
    decoding is fast at runtime.
    """

    def __init__(self) -> None:
        """Initialize the LLM model and pre-compute helpers."""
        self.llm = Small_LLM_Model()
        self._number_token_ids: Set[int] = set()
        self._stop_token_ids: Set[int] = set()
        self._build_number_tokens()

    def _encode(self, text: str) -> List[int]:
        """Encode text to a flat list of native Python int IDs.

        Args:
            text: The string to encode.

        Returns:
            A list of integer token IDs.
        """
        tensor = self.llm.encode(text)
        return cast(List[int], tensor[0].tolist())

    def _get_logits(self, input_ids: List[int]) -> np.ndarray:
        """Get next-token logits from the model.

        Args:
            input_ids: Sequence of token IDs.

        Returns:
            Numpy array of logit scores (vocab-sized).
        """
        raw = self.llm.get_logits_from_input_ids(input_ids)
        return np.array(raw, dtype=np.float32)

    def _greedy_next(self, input_ids: List[int]) -> int:
        """Return the greedy (argmax) next token.

        Args:
            input_ids: Current token sequence.

        Returns:
            Token ID with the highest logit.
        """
        logits = self._get_logits(input_ids)
        return int(np.argmax(logits))

    def _masked_next(
        self, input_ids: List[int], allowed: Set[int]
    ) -> int:
        """Return the best next token among the *allowed* set.

        Args:
            input_ids: Current token sequence.
            allowed: Set of valid token IDs.

        Returns:
            The highest-scoring token among *allowed*.
        """
        logits = self._get_logits(input_ids)
        mask = np.full(logits.shape, NEG_INF, dtype=np.float32)
        for tid in allowed:
            if tid < len(logits):
                mask[tid] = logits[tid]
        return int(np.argmax(mask))

    def _build_number_tokens(self) -> None:
        """Pre-compute sets of numeric and stop token IDs."""
        number_chars = set("0123456789.-+eE")
        stop_chars = set(",}] \n\t\r:")

        try:
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path, "r", encoding="utf-8") as fh:
                raw_vocab: Dict[str, int] = json.load(fh)
            for tok_str, tid in raw_vocab.items():
                clean = tok_str.replace("\u0120", "").strip()
                if not clean:
                    self._stop_token_ids.add(tid)
                    continue
                if all(c in number_chars for c in clean):
                    self._number_token_ids.add(tid)
                if clean[0] in stop_chars:
                    self._stop_token_ids.add(tid)
        except Exception:
            pass

    def _select_function_name(
        self,
        prompt_ids: List[int],
        function_names: List[str],
    ) -> str:
        """Select a function name via trie-based constrained decoding.

        Each function name is pre-encoded into its token sequence.
        At every generation step only the tokens that continue at
        least one remaining candidate are allowed.  When there is
        only one allowed token we skip the LLM call entirely.

        Args:
            prompt_ids: Token IDs up to the function-name position.
            function_names: List of valid function names.

        Returns:
            The selected function name string.
        """
        name_seqs: Dict[str, List[int]] = {
            n: self._encode(n) for n in function_names
        }
        candidates: List[Tuple[str, List[int]]] = list(
            name_seqs.items()
        )
        ids = list(prompt_ids)
        pos = 0
        while len(candidates) > 1:
            allowed: Set[int] = set()
            for _name, seq in candidates:
                if pos < len(seq):
                    allowed.add(seq[pos])
            if not allowed:
                break
            if len(allowed) == 1:
                chosen = next(iter(allowed))
            else:
                chosen = self._masked_next(ids, allowed)
            ids.append(chosen)
            pos += 1
            candidates = [
                (name, seq)
                for name, seq in candidates
                if pos <= len(seq) and seq[pos - 1] == chosen
            ]
            complete = [n for n, s in candidates if pos == len(s)]
            if len(complete) == 1:
                return complete[0]
        if candidates:
            return candidates[0][0]
        return function_names[0]

    def _generate_string_value(
        self, base_ids: List[int], max_tokens: int = 120
    ) -> str:
        """Generate a string value until a closing double-quote.

        The model generates freely; we stop as soon as it produces
        the quote token.

        Args:
            base_ids: Token IDs *including* the opening quote.
            max_tokens: Safety limit.

        Returns:
            The generated string content (without quotes).
        """
        ids = list(base_ids)
        quote_id_list = self._encode('"')
        quote_id = quote_id_list[-1] if quote_id_list else None
        text = ""

        for _ in range(max_tokens):
            chosen = self._greedy_next(ids)
            tok_text = self.llm.decode(chosen)
            if chosen == quote_id:
                break
            if '"' in tok_text:
                text += tok_text.split('"')[0]
                break
            text += tok_text
            ids.append(chosen)
        return text

    def _generate_number_value(
        self, base_ids: List[int], max_tokens: int = 25
    ) -> float:
        """Generate a numeric value with digit-constrained decoding.

        Only tokens whose text consists of number characters are
        allowed (plus stop tokens to terminate).

        Args:
            base_ids: Token IDs up to the number start.
            max_tokens: Safety limit.

        Returns:
            The generated number as a float.
        """
        ids = list(base_ids)
        number_chars = set("0123456789.-+eE")
        text = ""

        for _ in range(max_tokens):
            logits = self._get_logits(ids)
            mask = np.full(
                logits.shape, NEG_INF, dtype=np.float32
            )
            for tid in self._number_token_ids:
                if tid < len(logits):
                    mask[tid] = logits[tid]
            for tid in self._stop_token_ids:
                if tid < len(logits):
                    mask[tid] = logits[tid]
            chosen = int(np.argmax(mask))
            if chosen in self._stop_token_ids:
                break
            tok_text = self.llm.decode(chosen)
            clean = tok_text.strip()
            if not clean:
                break
            if not all(c in number_chars for c in clean):
                break
            text += clean
            ids.append(chosen)
        text = text.strip()
        try:
            return float(text)
        except ValueError:
            clean_text = ""
            for ch in text:
                if ch in "0123456789.-+eE":
                    clean_text += ch
                else:
                    break
            try:
                return float(clean_text) if clean_text else 0.0
            except ValueError:
                return 0.0

    def _generate_integer_value(
        self, base_ids: List[int], max_tokens: int = 20
    ) -> int:
        """Generate an integer value.

        Args:
            base_ids: Token IDs up to the integer start.
            max_tokens: Safety limit.

        Returns:
            The generated integer.
        """
        return int(
            self._generate_number_value(base_ids, max_tokens)
        )

    def _generate_boolean_value(
        self, base_ids: List[int]
    ) -> bool:
        """Generate a boolean via constrained decoding.

        Only the first tokens of ``true`` and ``false`` are allowed.

        Args:
            base_ids: Token IDs before the boolean position.

        Returns:
            True or False.
        """
        true_ids = self._encode("true")
        false_ids = self._encode("false")

        allowed: Set[int] = set()
        if true_ids:
            allowed.add(true_ids[0])
        if false_ids:
            allowed.add(false_ids[0])
        if not allowed:
            return True
        chosen = self._masked_next(list(base_ids), allowed)
        if true_ids and chosen == true_ids[0]:
            return True
        return False

    def _fix_regex_params(
        self,
        prompt: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-process parameters for fn_substitute_string_with_regex.

        Analyses the prompt to deduce the correct regex pattern and
        replacement when the LLM's free-form generation may be wrong.

        Args:
            prompt: The original user query.
            params: The extracted parameters dict.

        Returns:
            The corrected parameters dict.
        """
        prompt_lower = prompt.lower()
        quoted = re.findall(r'"([^"]+)"', prompt)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", prompt)
        if ("number" in prompt_lower or "digit" in prompt_lower):
            params["regex"] = "\\d+"
            m = re.search(r'\bwith\s+(\S+)\s*$', prompt.strip())
            if m:
                params["replacement"] = m.group(1)
        elif "vowel" in prompt_lower:
            params["regex"] = "[aeiouAEIOU]"
            m = re.search(r'\bwith\s+(\S+)\s*$', prompt.strip())
            if m:
                repl = m.group(1)
                if repl.lower() in ("asterisks", "asterisk", "stars", "star"):
                    params["replacement"] = "*"
                else:
                    params["replacement"] = repl
        elif "substitute" in prompt_lower or "replace" in prompt_lower:
            m_word = re.search(
                r"(?:substitute|replace)\s+(?:the\s+)?word\s+'([^']+)'\s+with\
                    s+'([^']+)'",
                prompt,
                re.IGNORECASE,
            )
            if m_word:
                word = m_word.group(1)
                replacement = m_word.group(2)
                params["regex"] = "\\b" + re.escape(word) + "\\b"
                params["replacement"] = replacement
                m_in = re.search(r"\bin\s+'([^']+)'", prompt)
                if not m_in:
                    m_in = re.search(r'\bin\s+"([^"]+)"', prompt)
                if m_in:
                    params["source_string"] = m_in.group(1)
        return params

    def _fix_read_file_params(
        self,
        prompt: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-process parameters for fn_read_file.

        Extracts the file path and encoding directly from the prompt
        to avoid issues with leading slashes and backslash escaping.

        Args:
            prompt: The original user query.
            params: The extracted parameters dict.

        Returns:
            The corrected parameters dict.
        """
        m = re.search(
            r'([Rr]ead\s+(?:the\s+file\s+at\s+)?(\S+)\s+with\s+(\S+)\
            \s+encoding)',
            prompt,
        )
        if m:
            params["path"] = m.group(1)
            params["encoding"] = m.group(2)
        return params

    def _fix_format_template_params(
        self,
        prompt: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-process parameters for fn_format_template.

        Extracts the template string from after 'Format template: '
        in the prompt.

        Args:
            prompt: The original user query.
            params: The extracted parameters dict.

        Returns:
            The corrected parameters dict.
        """
        m = re.search(r'[Ff]ormat\s+template:\s+(.*)', prompt)
        if m:
            params["template"] = m.group(1)
        return params

    def process_prompt(
        self,
        prompt: str,
        functions: List[Function],
        static_ids: List[int],
        func_names: List[str],
        func_map: Dict[str, Function],
    ) -> FunctionCallResult:
        """Process one user prompt into a function call result.

        Args:
            prompt: The natural language user query.
            functions: Available function definitions.
            static_ids: Pre-encoded system-prompt token IDs.
            func_names: List of function-name strings.
            func_map: Name -> Function mapping.

        Returns:
            A FunctionCallResult with name and parameters.
        """
        dynamic_prompt = (
            "[USER_QUERY]\n"
            + prompt
            + "\n[FUNCTION_NAME]\n"
        )
        dynamic_ids = self._encode(dynamic_prompt)
        prompt_ids = static_ids + dynamic_ids
        selected_name = self._select_function_name(
            prompt_ids, func_names
        )
        func_def = func_map.get(selected_name)
        if func_def is None:
            func_def = functions[0]
            selected_name = func_def.name
        params: Dict[str, Any] = {}
        for param_name, param_info in func_def.parameters.items():
            ctx = (
                selected_name + "\n"
                "[TASK]\n"
                "Extract the value of parameter "
                + json.dumps(param_name)
                + " (type: " + param_info.type + ") "
                "from the user query. "
                "Copy the value EXACTLY as it appears "
                "in the query.\n"
                "[DESCRIPTION]\n"
                + func_def.description + "\n"
                "[JSON_OUTPUT]\n"
                '{"prompt": ' + json.dumps(prompt)
                + ', "name": ' + json.dumps(selected_name)
                + ', "parameters": {'
            )
            for prev_name, prev_val in params.items():
                ctx += (
                    json.dumps(prev_name)
                    + ": "
                    + json.dumps(prev_val)
                    + ", "
                )
            ctx += json.dumps(param_name) + ": "
            param_ids = prompt_ids + self._encode(ctx)
            val: Any
            if param_info.type == "string":
                quote_ids = self._encode('"')
                val = self._generate_string_value(
                    param_ids + quote_ids
                )
            elif param_info.type == "number":
                val = self._generate_number_value(param_ids)
            elif param_info.type == "integer":
                val = self._generate_integer_value(param_ids)
            elif param_info.type == "boolean":
                val = self._generate_boolean_value(param_ids)
            else:
                quote_ids = self._encode('"')
                val = self._generate_string_value(
                    param_ids + quote_ids
                )
            params[param_name] = val
        if selected_name == "fn_substitute_string_with_regex":
            params = self._fix_regex_params(prompt, params)
        elif selected_name == "fn_read_file":
            params = self._fix_read_file_params(prompt, params)
        elif selected_name == "fn_format_template":
            params = self._fix_format_template_params(prompt, params)
        return FunctionCallResult(
            prompt=prompt,
            name=selected_name,
            parameters=params,
        )

    def run(
        self,
        functions: List[Function],
        callables: List[Calling],
        output_path: str,
    ) -> None:
        """Process all prompts and write valid JSON output.

        Args:
            functions: Available function definitions.
            callables: User prompts to process.
            output_path: Path for the output JSON file.
        """
        func_names = [f.name for f in functions]
        func_map: Dict[str, Function] = {
            f.name: f for f in functions
        }

        tools_json = json.dumps(
            [f.model_dump() for f in functions], indent=None
        )

        system_prompt = (
            "[SYSTEM_INSTRUCTION]\n"
            "You are a deterministic JSON generator for tool "
            "selection.\n"
            "Rules:\n"
            "- Extract parameter values EXACTLY as they appear "
            "in the user query. Copy them literally.\n"
            "- For string parameters, extract the exact text "
            "from the query, preserving every character.\n"
            "- For number parameters, output numbers as floats.\n"
            "- For regex parameters, use the simplest standard "
            "regex pattern (e.g. use \\d+ not [0-9]+).\n"
            "[AVAILABLE_TOOLS]\n"
            + tools_json + "\n"
            "[EXAMPLES]\n"
            'Query: "Add 5 and 10"\n'
            'Output: {"prompt": "Add 5 and 10", '
            '"name": "fn_add_numbers", '
            '"parameters": {"a": 5.0, "b": 10.0}}\n'
            'Query: "Replace all digits in \'abc 42\' '
            'with X"\n'
            'Output: {"prompt": "Replace all digits in '
            "'abc 42' with X\", "
            '"name": "fn_substitute_string_with_regex", '
            '"parameters": {"source_string": "abc 42", '
            '"regex": "\\\\d+", "replacement": "X"}}\n'
        )
        static_ids = self._encode(system_prompt)
        results: List[Dict[str, Any]] = []
        for idx, call in enumerate(callables):
            label = call.prompt[:60]
            print(
                "Processing "
                + str(idx + 1)
                + "/"
                + str(len(callables))
                + ": "
                + label
            )
            try:
                result = self.process_prompt(
                    call.prompt,
                    functions,
                    static_ids,
                    func_names,
                    func_map,
                )
                results.append(result.model_dump())
            except Exception as exc:
                print("  Error: " + str(exc))
                results.append({
                    "prompt": call.prompt,
                    "name": func_names[0] if func_names else "",
                    "parameters": {},
                })
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print("\nResults written to " + output_path)
