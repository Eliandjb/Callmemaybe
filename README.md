*This project has been created as part of the 42 curriculum by edjebbou.*

# 📞 Call Me Maybe

> *Translating natural language into structured function calls — using a tiny LLM that punches above its weight.*

---

## Table of Contents

- [Description](#description)
- [Algorithm Explained](#algorithm-explained)
- [Design Decisions](#design-decisions)
- [Installation & Usage](#installation--usage)
- [Example Usage](#example-usage)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced](#challenges-faced)
- [Testing Strategy](#testing-strategy)
- [Resources & AI Usage](#resources--ai-usage)

---

## Description

**Call Me Maybe** is a function-calling engine that parses natural language prompts and maps them to structured JSON function calls — using a **0.6-billion parameter language model** ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)).

The core challenge with small LLMs is their unreliability: left unconstrained, they hallucinate keys, misformat values, and generate structurally broken JSON. This project solves that problem with **constrained decoding**: a generation-time mechanism that intercepts the model's output logits and enforces a strict structural grammar, guaranteeing 100% syntactically valid JSON on every run.

The pipeline takes three inputs:

- A **function definitions file** — a JSON schema describing available functions, their parameters, and expected types.
- An **input file** — a list of natural language prompts.
- An **output file path** — where the structured function call results are written.

The model never "guesses" the format. Every token it emits is structurally legal, validated against the schema at generation time.

---

## Algorithm Explained

The engine implements a **constrained decoding** strategy built around two complementary mechanisms:

### 1. Trie-Based Function Name Enforcement

Before generation begins, all valid function names are loaded from the schema and inserted into a **trie** (prefix tree). During decoding, as the model generates characters for the function name field, the trie is traversed in lockstep. At each position, only the token IDs that correspond to valid continuations of the current prefix are permitted — all others have their logits set to `-inf`.

This guarantees the generated function name is always one of the predefined identifiers, with no possible hallucination.

### 2. Type-Aware Parameter Masking

For parameter values, the engine pre-computes **sets of allowed token IDs** per type at initialization time:

- **Integers / Floats**: only digit characters, a leading `-`, and a decimal point `.` are allowed.
- **Strings**: any token is permitted within the delimited string boundaries, but opening/closing quotes and escape sequences are strictly enforced.
- **Booleans / Nulls**: only the exact token sequences `true`, `false`, and `null` are allowed.

During generation, the engine tracks a **finite-state machine (FSM)** representing the current position within the JSON structure (e.g., "inside a string value", "parsing an integer argument"). At each decoding step:

1. The model computes a full vocabulary distribution (logits).
2. The engine determines which tokens are valid given the FSM state.
3. A **binary mask** is applied: invalid token logits are set to `-inf`.
4. The model samples from the masked distribution.

The result is a model that cannot structurally deviate from the expected output format.

```
Prompt
  │
  ▼
┌─────────────────────┐
│   Qwen3-0.6B LLM    │  ← produces raw logits at each step
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Constrained Mask   │  ← trie (names) + digit/type masks (params)
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   Valid JSON Output │  ← 100% structurally valid, schema-conformant
└─────────────────────┘
```

---

## Design Decisions

**Pre-computed token ID sets in `__init__`**
Mask generation is invoked at every single decoding step. To avoid re-tokenizing the full vocabulary on each call, all allowed token ID sets — digits, booleans, structural characters like `{`, `}`, `:`, `,` — are computed once during engine initialization and stored as frozen sets. This keeps per-step masking overhead negligible regardless of prompt length.

**Trie for function names instead of a flat allow-list**
A flat set lookup would require the full function name to be known before it can be validated, which is impossible mid-generation. The trie enables incremental validation: at every generated character, the engine checks whether the current prefix still leads to at least one valid function name. If not, that branch is pruned immediately. This also handles functions with shared prefixes cleanly.

**Finite-state machine for structural tracking**
Rather than post-processing the output, the FSM makes constraints dynamic and context-aware. The same token — for example a double quote `"` — carries entirely different structural meaning depending on whether we are opening a key, closing a value, or mid-string. The FSM disambiguates these states and applies the correct mask for each.

**Regex post-processing for natural language extraction**
The constrained decoder guarantees structural validity, but it still needs to know *which* function to call and *what* the argument values are before generation begins. A regex extraction layer handles this: it parses the raw natural language prompt, identifies the target function and its arguments, and feeds clean, disambiguated inputs to the decoder. This layer proved critical for handling edge cases — multi-sentence prompts, prompts containing quoted strings with JSON-like syntax, and implicit argument references.

**`uv` for dependency management**
`uv` was chosen over `pip` / `venv` for its significantly faster dependency resolution and fully deterministic lock files. This makes the project reproducible across the various 42 school workstations without environment drift.

---

## Installation & Usage

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager

### Install

```bash
make install
```

Resolves and installs all dependencies defined in `pyproject.toml` into an isolated `uv`-managed virtual environment.

### Run

```bash
make run
```

Or explicitly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Lint

```bash
make lint
```

Runs `flake8` and `mypy` in strict mode against the source tree.

---

## Example Usage

```bash
$ make run

Loading function schema from data/input/functions_definition.json...  [3 functions loaded]
Processing prompts from data/input/function_calling_tests.json...     [100 prompts]

[  1/100] "Book a flight to Tokyo on the 12th of March"
          → book_flight(destination="Tokyo", date="2024-03-12")  ✓

[  2/100] "Set a reminder for 7:30am tomorrow called Morning Standup"
          → set_reminder(time="07:30", label="Morning Standup")  ✓

[  3/100] "What is the weather in Lyon right now?"
          → get_weather(city="Lyon", unit="celsius")             ✓

...

[100/100] Done. (2m37s — avg 1.57s/prompt)

Results written to data/output/function_calling_results.json
```

**Output snippet (`function_calling_results.json`):**

```json
[
  {
    "prompt": "Book a flight to Tokyo on the 12th of March",
    "function_call": {
      "name": "book_flight",
      "arguments": {
        "destination": "Tokyo",
        "date": "2024-03-12"
      }
    }
  },
  {
    "prompt": "Set a reminder for 7:30am tomorrow called Morning Standup",
    "function_call": {
      "name": "set_reminder",
      "arguments": {
        "time": "07:30",
        "label": "Morning Standup"
      }
    }
  }
]
```

---

## Performance Analysis

| Metric | Result |
|---|---|
| JSON structural validity | **100%** (guaranteed by construction) |
| Argument extraction accuracy | **99%+** |
| Total processing time (100 prompts) | **~2 minutes 37 seconds** |
| Average time per prompt | **~1.57 seconds** |
| Hardware used | Standard 42 School Workstations |

The 100% structural validity guarantee is a hard property of the constrained decoder — it holds by construction regardless of prompt content, model confidence, or sampling temperature. It is not an empirical observation that could regress.

Argument accuracy at 99%+ reflects the robustness of the combined regex extraction and type-aware masking layers. The rare failures occur on highly ambiguous multi-clause prompts where the correct argument value is genuinely underspecified by the natural language input.

---

## Challenges Faced

**Satisfying `mypy` strict mode with the provided `llm_sdk`**
The school-provided `llm_sdk` module returned loosely typed tensors and generic dictionaries with no type stubs. Under `--warn-return-any` and `--disallow-untyped-defs`, every interaction with the SDK triggered a cascade of typing errors. The solution was disciplined use of `cast()` from the `typing` module to assert known types at SDK boundaries — without modifying any external school code. Getting this right required understanding exactly where `mypy`'s inference broke down and inserting the minimum number of casts to satisfy the checker without obscuring real type errors elsewhere.

**Building robust regex logic for natural language extraction**
Natural language is far messier than it looks at first glance. A prompt like *"remind me to call Sarah at 3pm"* and *"set a 3pm reminder to call Sarah"* encode the same intent with completely different syntax. Building regex patterns that reliably extracted function names and argument values across all input styles — including prompts with quoted strings, nested clauses, and implicit arguments — required significant iterative testing and careful handling of edge cases.

**Isolating the virtual environment correctly in the `Makefile`**
Getting `uv` to activate and use the correct isolated environment within `make` targets (which spawn subshells) required careful construction of the Makefile to ensure that `uv run` was consistently used as the execution wrapper rather than relying on shell `PATH` activation, which does not persist across `make` recipe lines.

---

## Testing Strategy

**`flake8` + `mypy` strict compliance as a first-class gate**
The primary correctness gate was the 42 moulinette's automated linting pipeline. Every iteration was validated against `flake8` (style) and `mypy --strict` (types) before being considered complete. This discipline caught a significant number of logic errors early — strict typing often surfaces incorrect assumptions about data shapes and optionality before they become runtime bugs.

**Iterative accuracy benchmarking against the provided test files**
The project's correctness was measured by running the full pipeline against the provided `function_calling_tests.json` and manually inspecting mismatches between the generated output and the expected results. Accuracy and runtime were tracked across iterations, with the 99%+ accuracy and 2m37s total time serving as the target benchmark.

**Edge case stress testing on the extraction layer**
The regex extraction layer was specifically tested against adversarial inputs: prompts containing embedded JSON-like syntax, prompts where the function name appeared in a quoted context, multi-sentence prompts where the actionable instruction was not the first clause, and prompts with argument values containing special characters or units.

---

## Resources & AI Usage

### Technical Documentation

- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Python `typing` module — `cast`](https://docs.python.org/3/library/typing.html#typing.cast)
- [uv — Python packaging and project management](https://docs.astral.sh/uv/)

### AI Usage

I used **Gemini** as an interactive pair-programming assistant for three specific problems: debugging highly restrictive `mypy` typing errors introduced by the school-provided `llm_sdk` module (particularly around `cast()` usage and `--warn-return-any` compliance), correctly isolating the virtual environment within the `Makefile` so that `uv` targets behaved consistently, and structuring the first draft of this README.

---

## Project Structure

```
call-me-maybe/
├── data/
│   ├── input/
│   │   ├── functions_definition.json   # Schema of callable functions
│   │   └── function_calling_tests.json # Natural language prompts
│   └── output/
│       └── function_calling_results.json
├── src/
│   ├── __main__.py                     # Entry point & argument parsing
│   ├── engine.py                       # Constrained decoding engine
│   ├── trie.py                         # Trie for function name enforcement
│   ├── masks.py                        # Token mask builders per type
│   └── schema.py                       # Function schema loading & validation
├── Makefile
├── pyproject.toml
└── README.md
```

---

*Made with too much coffee and a healthy distrust of unconstrained language models.*
