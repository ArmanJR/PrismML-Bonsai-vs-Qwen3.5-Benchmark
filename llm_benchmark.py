# /// script
# requires-python = ">=3.10"
# dependencies = ["openai"]
# ///
"""LLM Benchmark: Accuracy + Performance Evaluation for Jetson Orin models."""

import csv
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

log = logging.getLogger("llm_benchmark")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BASE_URL = "http://localhost:8001/v1"
HEALTH_URL = "http://localhost:8001/health"
API_KEY = "sk-no-key-required"
HEALTH_TIMEOUT_S = 120
CODE_EXEC_TIMEOUT_S = 5
N_RUNS = 3  # Number of runs per question for statistical reliability
SYSTEM_PROMPT = "You are a helpful assistant. Answer directly and concisely."

MODELS: list[dict[str, Any]] = [
    {"name": "Qwen3.5-35B-A3B", "service": "llama-server-qwen3.5-35b-a3b",
     "alias": "qwen3.5-35b-a3b", "params_b": 35.5, "quant": "Q4_K_M"},
    {"name": "Qwen3.6-27B", "service": "llama-server-qwen3.6-27b",
     "alias": "qwen3.6-27b", "params_b": 27.3, "quant": "Q4_K_M"},
    {"name": "Bonsai-27B", "service": "llama-server-bonsai-27b",
     "alias": "bonsai-27b", "params_b": 27.3, "quant": "Q1_0"},
    {"name": "Ternary-Bonsai-27B", "service": "llama-server-ternary-bonsai-27b",
     "alias": "ternary-bonsai-27b", "params_b": 27.3, "quant": "Q2_0"},
    {"name": "Qwen3.5-27B", "service": "llama-server-qwen3.5-27b",
     "alias": "qwen3.5-27b", "params_b": 26.9, "quant": "Q4_K_M"},
    {"name": "Qwen3.5-9B", "service": "llama-server-qwen3.5-9b",
     "alias": "qwen3.5-9b", "params_b": 8.95, "quant": "Q4_K_M"},
    {"name": "Bonsai-8B", "service": "llama-server-bonsai-8b",
     "alias": "bonsai-8b", "params_b": 8.19, "quant": "Q1_0"},
    {"name": "Ternary-Bonsai-8B", "service": "llama-server-ternary-bonsai-8b",
     "alias": "ternary-bonsai-8b", "params_b": 8.19, "quant": "mlx-2bit"},
    {"name": "Qwen3.5-4B", "service": "llama-server-qwen3.5-4b",
     "alias": "qwen3.5-4b", "params_b": 4.21, "quant": "Q4_K_M"},
    {"name": "Ternary-Bonsai-4B", "service": "llama-server-ternary-bonsai-4b",
     "alias": "ternary-bonsai-4b", "params_b": 4.02, "quant": "mlx-2bit"},
    {"name": "Qwen3.5-2B", "service": "llama-server-qwen3.5-2b",
     "alias": "qwen3.5-2b", "params_b": 1.89, "quant": "Q4_K_M"},
    {"name": "Ternary-Bonsai-1.7B", "service": "llama-server-ternary-bonsai-1.7b",
     "alias": "ternary-bonsai-1.7b", "params_b": 1.72, "quant": "mlx-2bit"},
    {"name": "Qwen3.5-0.8B", "service": "llama-server-qwen3.5-0.8b",
     "alias": "qwen3.5-0.8b", "params_b": 0.82, "quant": "Q4_K_S"},
]

CAT_GK = "general_knowledge"
CAT_MATH = "math"
CAT_CODE = "coding"
CAT_HIST = "history"
CAT_LOGIC = "logical_reasoning"
CAT_LANG = "language_understanding"
CAT_FA = "persian"

CATEGORIES = [CAT_GK, CAT_MATH, CAT_CODE, CAT_HIST, CAT_LOGIC, CAT_LANG, CAT_FA]

# ═══════════════════════════════════════════════════════════════════════════════
# Coding test harnesses
# ═══════════════════════════════════════════════════════════════════════════════

_FIZZBUZZ_TESTS = """\
_p = 0
try:
    assert fizzbuzz(1) == ["1"], f"got {fizzbuzz(1)}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert fizzbuzz(3) == ["1", "2", "Fizz"], f"got {fizzbuzz(3)}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"], f"got {fizzbuzz(5)}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = fizzbuzz(15)
    assert _r[-1] == "FizzBuzz" and len(_r) == 15, f"last={_r[-1]}, len={len(_r)}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_MAX3_TESTS = """\
_p = 0
try:
    assert max_of_three(1, 2, 3) == 3
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert max_of_three(3, 2, 1) == 3
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert max_of_three(1, 3, 2) == 3
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert max_of_three(-1, -2, -3) == -1
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_PALINDROME_TESTS = """\
_p = 0
try:
    assert is_palindrome("racecar") == True
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert is_palindrome("A man a plan a canal Panama") == True
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert is_palindrome("hello") == False
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert is_palindrome("Was it a car or a cat I saw?") == True
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_TWOSUM_TESTS = """\
_p = 0
try:
    _r = two_sum([2, 7, 11, 15], 9)
    assert set(_r) == {0, 1}, f"got {_r}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    _r = two_sum([3, 2, 4], 6)
    assert set(_r) == {1, 2}, f"got {_r}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    _r = two_sum([3, 3], 6)
    assert set(_r) == {0, 1}, f"got {_r}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
print(f"RESULT: {_p}/3")
"""

_VOWELS_TESTS = """\
_p = 0
try:
    _r = count_vowels("hello")
    assert _r == {"a": 0, "e": 1, "i": 0, "o": 1, "u": 0}, f"got {_r}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    _r = count_vowels("AEIOU")
    assert _r == {"a": 1, "e": 1, "i": 1, "o": 1, "u": 1}, f"got {_r}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    _r = count_vowels("xyz")
    assert _r == {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}, f"got {_r}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = count_vowels("aardvark")
    assert _r == {"a": 3, "e": 0, "i": 0, "o": 0, "u": 0}, f"got {_r}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_FLATTEN_TESTS = """\
_p = 0
try:
    assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert flatten([]) == []
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert flatten([[1, 2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert flatten([1, 2, 3]) == [1, 2, 3]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_BUGFIX_TESTS = """\
_p = 0
try:
    _r = merge_sorted([1, 3, 5], [2, 4, 6])
    assert _r == [1, 2, 3, 4, 5, 6], f"got {_r}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    _r = merge_sorted([], [1, 2, 3])
    assert _r == [1, 2, 3], f"got {_r}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    _r = merge_sorted([1, 1, 1], [1, 1])
    assert _r == [1, 1, 1, 1, 1], f"got {_r}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = merge_sorted([10], [])
    assert _r == [10], f"got {_r}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_GROUPBY_TESTS = """\
_p = 0
try:
    _r = group_by_key([("a", 1), ("b", 2), ("a", 3)])
    assert _r == {"a": [1, 3], "b": [2]}, f"got {_r}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    _r = group_by_key([])
    assert _r == {}, f"got {_r}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    _r = group_by_key([("x", 10), ("y", 20), ("x", 30), ("y", 40), ("z", 50)])
    assert _r == {"x": [10, 30], "y": [20, 40], "z": [50]}, f"got {_r}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = group_by_key([("a", 1)])
    assert _r == {"a": [1]}, f"got {_r}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_TOPO_SORT_TESTS = """\
_p = 0
try:
    _r = topological_sort({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []})
    # Verify it's a valid topological order
    _idx = {v: i for i, v in enumerate(_r)}
    assert set(_r) == {"a", "b", "c", "d"}, f"wrong nodes: {_r}"
    assert _idx["a"] < _idx["b"] and _idx["a"] < _idx["c"], f"a not before b,c: {_r}"
    assert _idx["b"] < _idx["d"] and _idx["c"] < _idx["d"], f"b,c not before d: {_r}"
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    _r = topological_sort({"x": []})
    assert _r == ["x"], f"got {_r}"
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    _r = topological_sort({})
    assert _r == [], f"got {_r}"
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = topological_sort({"a": ["b"], "b": ["c"], "c": []})
    assert _r == ["a", "b", "c"], f"got {_r}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_UNIQUE_ORDER_TESTS = """\
_p = 0
try:
    assert unique_preserve_order([3,1,3,2,1]) == [3,1,2]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert unique_preserve_order([]) == []
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert unique_preserve_order([1,1,1,1]) == [1]
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert unique_preserve_order([5,4,3,2,1]) == [5,4,3,2,1]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_CHUNK_STRING_TESTS = """\
_p = 0
try:
    assert chunk_string("abcdefgh", 3) == ["abc","def","gh"]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert chunk_string("abcdef", 3) == ["abc","def"]
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert chunk_string("a", 5) == ["a"]
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert chunk_string("abcdefghij", 4) == ["abcd","efgh","ij"]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_ROTATE_MATRIX_TESTS = """\
_p = 0
try:
    assert rotate_matrix_clockwise([[1,2,3],[4,5,6]]) == [[4,1],[5,2],[6,3]]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert rotate_matrix_clockwise([[1,2],[3,4]]) == [[3,1],[4,2]]
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert rotate_matrix_clockwise([[1]]) == [[1]]
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert rotate_matrix_clockwise([[1,2,3]]) == [[1],[2],[3]]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_DECODE_RANGES_TESTS = """\
_p = 0
try:
    assert decode_ranges("1-3,5,8-9") == [1,2,3,5,8,9]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert decode_ranges("5") == [5]
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert decode_ranges("1-5") == [1,2,3,4,5]
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert decode_ranges("10,20-22,30") == [10,20,21,22,30]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_NEST_DOTS_TESTS = """\
_p = 0
try:
    assert nest_by_dots({"a.b": 1, "a.c": 2, "d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert nest_by_dots({}) == {}
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert nest_by_dots({"x": 10}) == {"x": 10}
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    _r = nest_by_dots({"a.b.c": 1, "a.b.d": 2, "a.e": 3})
    assert _r == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}, f"got {_r}"
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

_INTERVAL_GAPS_TESTS = """\
_p = 0
try:
    assert interval_gaps([[1,3],[2,5],[7,8]], 0, 10) == [[0,1],[5,7],[8,10]]
    _p += 1; print("PASS 1")
except Exception as _e:
    print(f"FAIL 1: {_e}")
try:
    assert interval_gaps([], 0, 5) == [[0,5]]
    _p += 1; print("PASS 2")
except Exception as _e:
    print(f"FAIL 2: {_e}")
try:
    assert interval_gaps([[0,10]], 0, 10) == []
    _p += 1; print("PASS 3")
except Exception as _e:
    print(f"FAIL 3: {_e}")
try:
    assert interval_gaps([[2,4],[6,8]], 0, 10) == [[0,2],[4,6],[8,10]]
    _p += 1; print("PASS 4")
except Exception as _e:
    print(f"FAIL 4: {_e}")
print(f"RESULT: {_p}/4")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Questions (98 total — 7 categories x 14)
# ═══════════════════════════════════════════════════════════════════════════════

QUESTIONS: list[dict[str, Any]] = [
    # ── General Knowledge (1-8) ───────────────────────────────────────────
    {"id": 1, "category": CAT_GK, "difficulty": "easy",
     "prompt": ("What is the chemical symbol for gold?\n"
                "A) Au\nB) Ag\nC) Fe\nD) Gd\n"
                "Answer with just the letter."),
     "scoring_type": "exact_match", "expected": ["a", "au"],
     "max_tokens": 64},

    {"id": 2, "category": CAT_GK, "difficulty": "easy",
     "prompt": "What gas do plants absorb from the atmosphere for photosynthesis? Answer in one or two words.",
     "scoring_type": "exact_match", "expected": ["carbon dioxide", "co2"],
     "max_tokens": 64},

    {"id": 3, "category": CAT_GK, "difficulty": "medium",
     "prompt": "What is the tallest mountain in the world, and what is its height in meters?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["everest"], 0.5),
         (["8848", "8849", "8,848", "8,849"], 0.5),
     ], "max_tokens": 256},

    {"id": 4, "category": CAT_GK, "difficulty": "medium",
     "prompt": "Name the four fundamental forces of nature.",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["gravit"], 0.25),
         (["electromagnet"], 0.25),
         (["strong nuclear", "strong force", "strong interaction"], 0.25),
         (["weak nuclear", "weak force", "weak interaction"], 0.25),
     ], "max_tokens": 256},

    {"id": 5, "category": CAT_GK, "difficulty": "medium",
     "prompt": "Which organ in the human body produces insulin? Answer with just the organ name.",
     "scoring_type": "exact_match", "expected": ["pancreas"],
     "max_tokens": 64},

    {"id": 6, "category": CAT_GK, "difficulty": "hard",
     "prompt": ("What phenomenon explains why the sky appears blue during the day? "
                "Name the type of scattering responsible and explain why blue light "
                "is affected more than red."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["rayleigh"], 0.5),
         (["wavelength", "shorter", "frequency", "inverse"], 0.5),
     ], "max_tokens": 512},

    {"id": 7, "category": CAT_GK, "difficulty": "hard",
     "prompt": "What is the Haber-Bosch process, and why is it significant?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["ammonia"], 0.5),
         (["fertilizer", "fertiliser", "agriculture", "food production"], 0.5),
     ], "max_tokens": 512},

    {"id": 8, "category": CAT_GK, "difficulty": "hard",
     "prompt": "What is the Chandrasekhar limit, and what is its approximate value in solar masses?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["white dwarf"], 0.5),
         (["1.4", "1.44"], 0.5),
     ], "max_tokens": 512},

    # ── Mathematics (9-16) ────────────────────────────────────────────────
    {"id": 9, "category": CAT_MATH, "difficulty": "easy",
     "prompt": "What is 247 multiplied by 83? Give only the number.",
     "scoring_type": "exact_match", "expected": ["20501"],
     "max_tokens": 64},

    {"id": 10, "category": CAT_MATH, "difficulty": "easy",
     "prompt": "What is 15% of 240? Give only the number.",
     "scoring_type": "exact_match", "expected": ["36"],
     "max_tokens": 64},

    {"id": 11, "category": CAT_MATH, "difficulty": "medium",
     "prompt": "Solve for x: 3x + 7 = 34. Give only the value of x.",
     "scoring_type": "exact_match", "expected": ["9"],
     "max_tokens": 64},

    {"id": 12, "category": CAT_MATH, "difficulty": "medium",
     "prompt": ("A store offers a 20% discount on a $150 item, then an additional "
                "10% off the already discounted price. What is the final price in "
                "dollars? Give only the number."),
     "scoring_type": "exact_match", "expected": ["108"],
     "max_tokens": 256},

    {"id": 13, "category": CAT_MATH, "difficulty": "medium",
     "prompt": "What is the sum of the interior angles of a hexagon, in degrees? Give only the number.",
     "scoring_type": "exact_match", "expected": ["720"],
     "max_tokens": 64},

    {"id": 14, "category": CAT_MATH, "difficulty": "hard",
     "prompt": "What is C(10,3), that is, '10 choose 3'? Give only the number.",
     "scoring_type": "exact_match", "expected": ["120"],
     "max_tokens": 64},

    {"id": 15, "category": CAT_MATH, "difficulty": "hard",
     "prompt": ("A car travels the first half of a 240 km journey at 40 km/h "
                "and the second half at 60 km/h. What is the average speed for "
                "the entire journey in km/h? Give only the number."),
     "scoring_type": "exact_match", "expected": ["48"],
     "max_tokens": 256},

    {"id": 16, "category": CAT_MATH, "difficulty": "hard",
     "prompt": "What is the definite integral of x² from 0 to 3? Give only the number.",
     "scoring_type": "exact_match", "expected": ["9"],
     "max_tokens": 256},

    # ── Coding (17-24) ────────────────────────────────────────────────────
    {"id": 17, "category": CAT_CODE, "difficulty": "easy",
     "prompt": ("Write a Python function `fizzbuzz(n)` that takes an integer n and "
                "returns a list of strings from 1 to n. For multiples of 3, use "
                "'Fizz'. For multiples of 5, use 'Buzz'. For multiples of both 3 "
                "and 5, use 'FizzBuzz'. Otherwise, use the string representation "
                "of the number. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "fizzbuzz",
     "test_code": _FIZZBUZZ_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 18, "category": CAT_CODE, "difficulty": "easy",
     "prompt": ("Write a Python function `max_of_three(a, b, c)` that returns the "
                "largest of three numbers. Do not use the built-in max() function. "
                "Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "max_of_three",
     "test_code": _MAX3_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 19, "category": CAT_CODE, "difficulty": "medium",
     "prompt": ("Write a Python function `is_palindrome(s)` that returns True if "
                "the string is a palindrome, ignoring case, spaces, and punctuation. "
                "Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "is_palindrome",
     "test_code": _PALINDROME_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 20, "category": CAT_CODE, "difficulty": "medium",
     "prompt": ("Write a Python function `two_sum(nums, target)` that takes a list "
                "of integers and a target sum, and returns a list of two indices "
                "whose values add up to the target. Each input has exactly one "
                "solution; you may not use the same element twice. Return only the "
                "Python code."),
     "scoring_type": "code_exec", "function_name": "two_sum",
     "test_code": _TWOSUM_TESTS, "num_tests": 3, "max_tokens": 1024},

    {"id": 21, "category": CAT_CODE, "difficulty": "medium",
     "prompt": ("The following Python function is supposed to merge two sorted lists "
                "into one sorted list, but it has bugs. Fix all the bugs.\n\n"
                "```python\n"
                "def merge_sorted(a, b):\n"
                "    result = []\n"
                "    i = j = 0\n"
                "    while i < len(a) and j < len(b):\n"
                "        if a[i] <= b[j]:\n"
                "            result.append(a[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            result.append(b[j])\n"
                "            i += 1\n"
                "    return result\n"
                "```\n"
                "Return only the corrected Python code."),
     "scoring_type": "code_exec", "function_name": "merge_sorted",
     "test_code": _BUGFIX_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 22, "category": CAT_CODE, "difficulty": "hard",
     "prompt": ("Write a Python function `flatten(lst)` that takes an arbitrarily "
                "nested list of integers and returns a flat list of all integers in "
                "order. Example: flatten([1, [2, [3, 4], 5], 6]) returns "
                "[1, 2, 3, 4, 5, 6]. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "flatten",
     "test_code": _FLATTEN_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 23, "category": CAT_CODE, "difficulty": "hard",
     "prompt": ("Write a Python function `group_by_key(pairs)` that takes a list of "
                "(key, value) tuples and returns a dictionary mapping each key to the "
                "list of its values, preserving order. Use only the Python standard "
                "library (collections is allowed). "
                "Example: group_by_key([('a', 1), ('b', 2), ('a', 3)]) returns "
                "{'a': [1, 3], 'b': [2]}. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "group_by_key",
     "test_code": _GROUPBY_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 24, "category": CAT_CODE, "difficulty": "hard",
     "prompt": ("Write a Python function `topological_sort(graph)` that takes a "
                "directed acyclic graph represented as a dictionary mapping each node "
                "(string) to a list of nodes it must come before (its dependents), "
                "and returns a list of nodes in valid topological order. "
                "Example: topological_sort({'a': ['b', 'c'], 'b': ['d'], "
                "'c': ['d'], 'd': []}) could return ['a', 'b', 'c', 'd'] or "
                "['a', 'c', 'b', 'd']. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "topological_sort",
     "test_code": _TOPO_SORT_TESTS, "num_tests": 4, "max_tokens": 1024},

    # ── History (25-32) ───────────────────────────────────────────────────
    {"id": 25, "category": CAT_HIST, "difficulty": "easy",
     "prompt": ("In what year did World War II end?\n"
                "A) 1943\nB) 1944\nC) 1945\nD) 1946\n"
                "Answer with just the letter."),
     "scoring_type": "exact_match", "expected": ["c", "1945"],
     "max_tokens": 64},

    {"id": 26, "category": CAT_HIST, "difficulty": "easy",
     "prompt": "What ancient civilization built the pyramids at Giza? Answer in one or two words.",
     "scoring_type": "exact_match",
     "expected": ["ancient egypt", "ancient egyptian", "ancient egyptians", "egyptians", "egyptian"],
     "max_tokens": 64},

    {"id": 27, "category": CAT_HIST, "difficulty": "medium",
     "prompt": "Who was the first President of the United States, and in what year did they take office?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["washington"], 0.5),
         (["1789"], 0.5),
     ], "max_tokens": 256},

    {"id": 28, "category": CAT_HIST, "difficulty": "medium",
     "prompt": "What event is commonly considered the start of the French Revolution, and in what year did it occur?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["bastille"], 0.5),
         (["1789"], 0.5),
     ], "max_tokens": 256},

    {"id": 29, "category": CAT_HIST, "difficulty": "medium",
     "prompt": "Who invented the printing press in Europe, and in approximately what year?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["gutenberg"], 0.5),
         (["1440", "1450", "1439", "1441", "1445", "1448"], 0.5),
     ], "max_tokens": 256},

    {"id": 30, "category": CAT_HIST, "difficulty": "hard",
     "prompt": "Name the three countries that formed the Triple Entente before World War I.",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["france", "french"], 0.34),
         (["britain", "british", "united kingdom", "uk", "england"], 0.33),
         (["russia", "russian"], 0.33),
     ], "max_tokens": 256},

    {"id": 31, "category": CAT_HIST, "difficulty": "hard",
     "prompt": "What was the significance of the Treaty of Westphalia (1648)?",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["sovereign"], 0.5),
         (["thirty years", "30 years", "thirty-years"], 0.5),
     ], "max_tokens": 512},

    {"id": 32, "category": CAT_HIST, "difficulty": "hard",
     "prompt": ("What was the Silk Road? Which two regions did it primarily connect? "
                "Name one major commodity besides silk that was traded on it."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["trade", "trading", "commerce", "commercial"], 0.25),
         (["china", "chinese", "east asia", "han"], 0.25),
         (["europe", "european", "rome", "roman", "mediterranean"], 0.25),
         (["spice", "porcelain", "gold", "paper", "tea", "horse", "jade",
           "ivory", "gem", "textile", "glass", "silver", "ceramic"], 0.25),
     ], "max_tokens": 512},

    # ── Logical Reasoning (33-40) ─────────────────────────────────────────
    {"id": 33, "category": CAT_LOGIC, "difficulty": "easy",
     "prompt": ("All cats are animals. Some animals are pets. "
                "Can we conclude that all cats are pets? Answer only Yes or No."),
     "scoring_type": "exact_match", "expected": ["no"],
     "max_tokens": 64},

    {"id": 34, "category": CAT_LOGIC, "difficulty": "easy",
     "prompt": "What is the next number in the sequence: 2, 6, 18, 54, ? Answer with just the number.",
     "scoring_type": "exact_match", "expected": ["162"],
     "max_tokens": 64},

    {"id": 35, "category": CAT_LOGIC, "difficulty": "medium",
     "prompt": ("If it rains, the ground gets wet. The ground is wet. "
                "Can we conclude that it rained? Answer only Yes or No."),
     "scoring_type": "exact_match", "expected": ["no"],
     "max_tokens": 64},

    {"id": 36, "category": CAT_LOGIC, "difficulty": "medium",
     "prompt": ("If A > B and B > C, and C = 5, what is the minimum possible "
                "integer value of A? Answer with just the number."),
     "scoring_type": "exact_match", "expected": ["7"],
     "max_tokens": 256},

    {"id": 37, "category": CAT_LOGIC, "difficulty": "medium",
     "prompt": ("Alice, Bob, and Charlie each own exactly one pet: a cat, a dog, "
                "and a fish (one pet per person, no repeats). Given these clues:\n"
                "1. Alice does not own the cat.\n"
                "2. Bob does not own the dog.\n"
                "3. Charlie does not own the fish.\n"
                "4. Alice does not own the dog.\n"
                "Who owns which pet? Answer in the format: "
                "Alice=[pet], Bob=[pet], Charlie=[pet]"),
     "scoring_type": "constraint", "verifier": "verify_pet_puzzle",
     "max_tokens": 512},

    {"id": 38, "category": CAT_LOGIC, "difficulty": "hard",
     "prompt": ("Five houses stand in a row, numbered 1 to 5 from left to right. "
                "Each house is a different color: Red, Blue, Green, Yellow, White. "
                "Determine the color of each house given these constraints:\n"
                "1. The red house is immediately to the left of the blue house.\n"
                "2. The green house is at position 3.\n"
                "3. The yellow house is at one of the ends (position 1 or 5).\n"
                "4. The white house is not adjacent to the red house.\n"
                "5. The blue house is not at either end.\n"
                "List the colors from position 1 to 5, separated by commas."),
     "scoring_type": "constraint", "verifier": "verify_five_houses",
     "max_tokens": 512},

    {"id": 39, "category": CAT_LOGIC, "difficulty": "hard",
     "prompt": ("All roses are flowers. Some flowers fade quickly. "
                "Can we conclude that some roses fade quickly? Answer only Yes or No."),
     "scoring_type": "exact_match", "expected": ["no"],
     "max_tokens": 64},

    {"id": 40, "category": CAT_LOGIC, "difficulty": "hard",
     "prompt": ("A says 'B always lies.' B says 'A and I both tell the truth.' "
                "If exactly one of them is a truth-teller, who is it? "
                "Answer with just A or B."),
     "scoring_type": "exact_match", "expected": ["a"],
     "max_tokens": 256},

    # ── Language Understanding (41-48) ────────────────────────────────────
    {"id": 41, "category": CAT_LANG, "difficulty": "easy",
     "prompt": ("What is the sentiment of the following text?\n"
                "'I just watched that movie and it was absolutely terrible. "
                "The acting was wooden and the plot made no sense.'\n"
                "Answer with one word: Positive, Negative, or Neutral."),
     "scoring_type": "exact_match", "expected": ["negative"],
     "max_tokens": 64},

    {"id": 42, "category": CAT_LANG, "difficulty": "easy",
     "prompt": "Complete the analogy: Painter is to Canvas as Author is to ___. Answer with one word.",
     "scoring_type": "exact_match",
     "expected": ["book", "page", "paper", "manuscript", "novel", "story"],
     "max_tokens": 64},

    {"id": 43, "category": CAT_LANG, "difficulty": "medium",
     "prompt": ("Read the following passage and provide a one-sentence summary:\n\n"
                "'The Industrial Revolution, which began in Britain in the late 18th "
                "century, marked a major turning point in human history. It saw a "
                "shift from agrarian economies to industrial manufacturing, powered "
                "by innovations like the steam engine and mechanized textile "
                "production. This transformation led to urbanization, new social "
                "classes, and significant changes in working conditions. While it "
                "brought unprecedented economic growth, it also created environmental "
                "challenges and social inequalities that persist to this day.'\n\n"
                "Write your summary in one sentence."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["industrial revolution"], 0.35),
         (["manufactur", "industr", "factory", "machine", "mechaniz",
           "agrarian", "steam", "production"], 0.35),
         (["economic", "social", "urban", "transform", "change", "shift",
           "society", "growth", "inequality", "working"], 0.30),
     ], "max_tokens": 512},

    {"id": 44, "category": CAT_LANG, "difficulty": "medium",
     "prompt": ("Which meaning of the word 'bank' is used in this sentence?\n"
                "'We sat on the bank and watched the wildflowers sway in the breeze.'\n"
                "A) A financial institution\n"
                "B) The side of a river\n"
                "C) To tilt an aircraft\n"
                "D) To rely on something\n"
                "Answer with just the letter."),
     "scoring_type": "exact_match", "expected": ["b"],
     "max_tokens": 64},

    {"id": 45, "category": CAT_LANG, "difficulty": "medium",
     "prompt": "What is the antonym of 'ephemeral'? Answer with one word.",
     "scoring_type": "exact_match",
     "expected": ["permanent", "eternal", "lasting", "enduring", "perpetual", "everlasting"],
     "max_tokens": 64},

    {"id": 46, "category": CAT_LANG, "difficulty": "hard",
     "prompt": ("A colleague asks you: 'Do you think John would make a good "
                "manager?' You reply: 'Well, he's always on time and his desk is "
                "very organized.' What are you implying about John's management "
                "abilities?"),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["not", "lack", "poor", "weak", "doubt", "faint praise",
           "damning", "inadequate", "avoid", "deflect", "imply",
           "indirect", "negative", "unsuitable", "unqualified",
           "insufficient", "doesn't", "without"], 0.5),
         (["manage", "leader", "leadership", "managerial",
           "qualities", "skills", "ability", "competence"], 0.5),
     ], "max_tokens": 512},

    {"id": 47, "category": CAT_LANG, "difficulty": "hard",
     "prompt": ("Identify the literary device in this sentence: "
                "'The wind whispered through the trees.' Answer with one word."),
     "scoring_type": "exact_match", "expected": ["personification"],
     "max_tokens": 64},

    {"id": 48, "category": CAT_LANG, "difficulty": "hard",
     "prompt": ("Rewrite the following sentence from passive voice to active voice: "
                "'The cake was eaten by the children.'"),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["children ate", "child ate", "kids ate"], 0.5),
         (["ate the cake", "ate cake", "ate a cake"], 0.5),
     ], "max_tokens": 256},

    # ── Persian Language Understanding (49-56) ────────────────────────────
    {"id": 49, "category": CAT_FA, "difficulty": "easy",
     "prompt": "پایتخت ایران کجاست؟ فقط نام شهر را بنویسید.",
     "scoring_type": "exact_match",
     "expected": ["تهران", "tehran", "tehran"],
     "max_tokens": 64},

    {"id": 50, "category": CAT_FA, "difficulty": "easy",
     "prompt": "معنی کلمه «آسمان» به انگلیسی چیست؟ فقط یک کلمه بنویسید.",
     "scoring_type": "exact_match", "expected": ["sky"],
     "max_tokens": 64},

    {"id": 51, "category": CAT_FA, "difficulty": "medium",
     "prompt": "جمله زیر را به انگلیسی ترجمه کنید: «من یک دانشجو هستم.»",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["i "], 0.34),
         (["am", "'m"], 0.33),
         (["student"], 0.33),
     ], "max_tokens": 256},

    {"id": 52, "category": CAT_FA, "difficulty": "medium",
     "prompt": ("کدام فعل در این جمله استفاده شده است؟ "
                "«بچه\u200cها در پارک بازی می\u200cکنند.» فقط فعل را بنویسید."),
     "scoring_type": "exact_match",
     "expected": ["بازی می\u200cکنند", "بازی کردن", "بازی می کنند",
                  "play", "بازی میکنند", "می\u200cکنند", "میکنند",
                  "می کنند", "بازی"],
     "max_tokens": 64},

    {"id": 53, "category": CAT_FA, "difficulty": "hard",
     "prompt": ("این ضرب\u200cالمثل فارسی «تا تنور داغ است نان را بچسبان» "
                "معادل کدام ضرب\u200cالمثل انگلیسی است؟ فقط ضرب\u200cالمثل "
                "انگلیسی را بنویسید."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["strike while the iron is hot", "make hay while the sun shines",
           "seize the day", "carpe diem", "strike while", "make hay",
           "iron is hot", "opportunity", "seize"], 1.0),
     ], "max_tokens": 256},

    {"id": 54, "category": CAT_FA, "difficulty": "medium",
     "prompt": "جمله زیر را به فارسی ترجمه کنید: «The sun rises in the east.»",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["خورشید", "آفتاب"], 0.5),
         (["شرق", "مشرق"], 0.5),
     ], "max_tokens": 256},

    {"id": 55, "category": CAT_FA, "difficulty": "hard",
     "prompt": ("شعر زیر از کیست؟ "
                "«بنی\u200cآدم اعضای یکدیگرند / که در آفرینش ز یک گوهرند» "
                "فقط نام شاعر را بنویسید."),
     "scoring_type": "exact_match",
     "expected": ["سعدی", "saadi", "sa'di", "sadi", "sa`di"],
     "max_tokens": 64},

    {"id": 56, "category": CAT_FA, "difficulty": "hard",
     "prompt": ("در ادبیات فارسی، «شاهنامه» اثر کیست و موضوع اصلی آن چیست؟"),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["فردوسی", "ferdowsi", "firdawsi", "firdausi", "firdousi", "ferdosi"], 0.5),
         (["myth", "epic", "king", "hero", "histor", "legend",
           "حماسه", "پادشاه", "اسطوره", "تاریخ", "شاهان",
           "iran", "persia", "ایران"], 0.5),
     ], "max_tokens": 512},

    # ── General Knowledge (57-62) ────────────────────────────────────────
    {"id": 57, "category": CAT_GK, "difficulty": "easy",
     "prompt": "What is the SI unit of electric current? Answer with one word.",
     "scoring_type": "exact_match", "expected": ["ampere", "amp"],
     "max_tokens": 64},

    {"id": 58, "category": CAT_GK, "difficulty": "easy",
     "prompt": "Which metal is liquid at standard room temperature? Answer with one word.",
     "scoring_type": "exact_match", "expected": ["mercury"],
     "max_tokens": 64},

    {"id": 59, "category": CAT_GK, "difficulty": "medium",
     "prompt": "What mineral ore is the primary commercial source of aluminum? Answer with one word.",
     "scoring_type": "exact_match", "expected": ["bauxite"],
     "max_tokens": 64},

    {"id": 60, "category": CAT_GK, "difficulty": "medium",
     "prompt": "In which layer of Earth's atmosphere is most ozone found? Answer with one word.",
     "scoring_type": "exact_match", "expected": ["stratosphere"],
     "max_tokens": 64},

    {"id": 61, "category": CAT_GK, "difficulty": "hard",
     "prompt": ("What scientific law states that pressure applied to a confined fluid "
                "is transmitted undiminished in all directions? Answer with the law's name."),
     "scoring_type": "exact_match",
     "expected": ["pascal's law", "pascal's principle", "pascal"],
     "max_tokens": 128},

    {"id": 62, "category": CAT_GK, "difficulty": "hard",
     "prompt": ("What term describes the symbiotic association between fungi and "
                "plant roots? Answer with one word."),
     "scoring_type": "exact_match",
     "expected": ["mycorrhiza", "mycorrhizae", "mycorrhizal"],
     "max_tokens": 64},

    # ── Mathematics (63-68) ──────────────────────────────────────────────
    {"id": 63, "category": CAT_MATH, "difficulty": "easy",
     "prompt": "What is (18² − 17²)? Give only the number.",
     "scoring_type": "exact_match", "expected": ["35"],
     "max_tokens": 64},

    {"id": 64, "category": CAT_MATH, "difficulty": "easy",
     "prompt": ("A recipe uses 3 cups of flour for 8 muffins. How many cups are "
                "needed for 20 muffins? Give only the number."),
     "scoring_type": "exact_match", "expected": ["7.5", "15/2"],
     "max_tokens": 128},

    {"id": 65, "category": CAT_MATH, "difficulty": "medium",
     "prompt": "Solve for x: 2(x − 3) = 3x + 5. Give only the value of x.",
     "scoring_type": "exact_match", "expected": ["-11"],
     "max_tokens": 128},

    {"id": 66, "category": CAT_MATH, "difficulty": "medium",
     "prompt": ("At 3:40, what is the smaller angle between the hour hand and the "
                "minute hand of a clock? Give only the number of degrees."),
     "scoring_type": "exact_match", "expected": ["130"],
     "max_tokens": 128},

    {"id": 67, "category": CAT_MATH, "difficulty": "hard",
     "prompt": ("A tank is 3/5 full. After adding 24 liters, it becomes 3/4 full. "
                "What is the tank's total capacity in liters? Give only the number."),
     "scoring_type": "exact_match", "expected": ["160"],
     "max_tokens": 256},

    {"id": 68, "category": CAT_MATH, "difficulty": "hard",
     "prompt": ("Find the sum of the infinite geometric series: "
                "12 − 6 + 3 − 1.5 + … Give only the number."),
     "scoring_type": "exact_match", "expected": ["8"],
     "max_tokens": 256},

    # ── Coding (69-74) ───────────────────────────────────────────────────
    {"id": 69, "category": CAT_CODE, "difficulty": "easy",
     "prompt": ("Write a Python function `unique_preserve_order(items)` that returns "
                "a new list containing the first occurrence of each item in the "
                "original order. Example: unique_preserve_order([3,1,3,2,1]) -> "
                "[3,1,2]. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "unique_preserve_order",
     "test_code": _UNIQUE_ORDER_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 70, "category": CAT_CODE, "difficulty": "easy",
     "prompt": ("Write a Python function `chunk_string(s, n)` that splits a string "
                "into consecutive substrings of length n. The last chunk may be "
                "shorter. Example: chunk_string(\"abcdefgh\", 3) -> "
                "[\"abc\",\"def\",\"gh\"]. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "chunk_string",
     "test_code": _CHUNK_STRING_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 71, "category": CAT_CODE, "difficulty": "medium",
     "prompt": ("Write a Python function `rotate_matrix_clockwise(M)` that returns "
                "a new matrix rotated 90 degrees clockwise. Example: "
                "[[1,2,3],[4,5,6]] -> [[4,1],[5,2],[6,3]]. "
                "Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "rotate_matrix_clockwise",
     "test_code": _ROTATE_MATRIX_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 72, "category": CAT_CODE, "difficulty": "medium",
     "prompt": ("Write a Python function `decode_ranges(spec)` that takes a string "
                "like \"1-3,5,8-9\" and returns the list [1,2,3,5,8,9]. Assume the "
                "input contains only positive integers and inclusive ranges. "
                "Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "decode_ranges",
     "test_code": _DECODE_RANGES_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 73, "category": CAT_CODE, "difficulty": "hard",
     "prompt": ("Write a Python function `nest_by_dots(flat)` that converts a "
                "dictionary with dotted keys into a nested dictionary. Example: "
                "{\"a.b\": 1, \"a.c\": 2, \"d\": 3} should become "
                "{\"a\": {\"b\": 1, \"c\": 2}, \"d\": 3}. "
                "Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "nest_by_dots",
     "test_code": _NEST_DOTS_TESTS, "num_tests": 4, "max_tokens": 1024},

    {"id": 74, "category": CAT_CODE, "difficulty": "hard",
     "prompt": ("Write a Python function `interval_gaps(intervals, start, end)` "
                "where `intervals` is a list of occupied half-open intervals [a, b) "
                "inside the overall half-open interval [start, end). Return the "
                "uncovered gaps as a sorted list of merged half-open intervals. "
                "Example: interval_gaps([[1,3],[2,5],[7,8]], 0, 10) -> "
                "[[0,1],[5,7],[8,10]]. Return only the Python code."),
     "scoring_type": "code_exec", "function_name": "interval_gaps",
     "test_code": _INTERVAL_GAPS_TESTS, "num_tests": 4, "max_tokens": 1024},

    # ── History (75-80) ──────────────────────────────────────────────────
    {"id": 75, "category": CAT_HIST, "difficulty": "easy",
     "prompt": "Which empire built Machu Picchu? Answer in one or two words.",
     "scoring_type": "exact_match",
     "expected": ["inca", "inca empire", "incan"],
     "max_tokens": 64},

    {"id": 76, "category": CAT_HIST, "difficulty": "easy",
     "prompt": ("What structure's fall in 1989 became a symbol of the end of the "
                "division between East and West Berlin? Answer in one or two words."),
     "scoring_type": "exact_match",
     "expected": ["berlin wall", "the berlin wall"],
     "max_tokens": 64},

    {"id": 77, "category": CAT_HIST, "difficulty": "medium",
     "prompt": ("What document signed in 1215 is famous for limiting the power "
                "of the English king? Answer in one or two words."),
     "scoring_type": "exact_match",
     "expected": ["magna carta", "the magna carta"],
     "max_tokens": 64},

    {"id": 78, "category": CAT_HIST, "difficulty": "medium",
     "prompt": ("Which Roman emperor established Constantinople as an imperial "
                "capital? Answer with the ruler's name."),
     "scoring_type": "exact_match",
     "expected": ["constantine", "constantine i", "constantine the great"],
     "max_tokens": 128},

    {"id": 79, "category": CAT_HIST, "difficulty": "hard",
     "prompt": ("Which treaty signed in 1713 helped end the War of the Spanish "
                "Succession? Answer with the treaty's name."),
     "scoring_type": "exact_match",
     "expected": ["treaty of utrecht", "utrecht"],
     "max_tokens": 128},

    {"id": 80, "category": CAT_HIST, "difficulty": "hard",
     "prompt": ("Which Chinese dynasty began ruling after the fall of the Yuan "
                "dynasty in 1368? Answer in one or two words."),
     "scoring_type": "exact_match",
     "expected": ["ming", "ming dynasty", "the ming dynasty"],
     "max_tokens": 64},

    # ── Logical Reasoning (81-86) ────────────────────────────────────────
    {"id": 81, "category": CAT_LOGIC, "difficulty": "easy",
     "prompt": ("All metals conduct electricity. Copper is a metal. "
                "Can we conclude that copper conducts electricity? "
                "Answer only Yes or No."),
     "scoring_type": "exact_match", "expected": ["yes"],
     "max_tokens": 64},

    {"id": 82, "category": CAT_LOGIC, "difficulty": "easy",
     "prompt": ("What is the next number in the sequence: 1, 2, 4, 7, 11, ? "
                "Answer with just the number."),
     "scoring_type": "exact_match", "expected": ["16"],
     "max_tokens": 64},

    {"id": 83, "category": CAT_LOGIC, "difficulty": "medium",
     "prompt": ("No musicians dislike rhythm. Some drummers are musicians. "
                "Can we conclude that some drummers do not dislike rhythm? "
                "Answer only Yes or No."),
     "scoring_type": "exact_match", "expected": ["yes"],
     "max_tokens": 64},

    {"id": 84, "category": CAT_LOGIC, "difficulty": "medium",
     "prompt": ("Nadia, Omid, and Parisa each have exactly one different badge "
                "color: red, blue, and green.\nClues:\n"
                "1. Nadia does not have the red badge.\n"
                "2. Omid does not have the green badge.\n"
                "3. Parisa does not have the blue badge.\n"
                "4. Omid does not have the red badge.\n"
                "Who has which badge color? Answer in the format: "
                "Nadia=[color], Omid=[color], Parisa=[color]"),
     "scoring_type": "constraint", "verifier": "verify_badge_puzzle",
     "max_tokens": 512},

    {"id": 85, "category": CAT_LOGIC, "difficulty": "hard",
     "prompt": ("Ari, Bina, Cyrus, and Darya finish a race in positions 1 through "
                "4, with no ties.\nClues:\n"
                "1. Ari finished before Bina.\n"
                "2. Cyrus did not finish first or last.\n"
                "3. Darya finished immediately after Ari.\n"
                "4. Bina did not finish second.\n"
                "What is the finishing order from 1st to 4th? "
                "Answer as four names separated by commas."),
     "scoring_type": "constraint", "verifier": "verify_race_order",
     "max_tokens": 512},

    {"id": 86, "category": CAT_LOGIC, "difficulty": "hard",
     "prompt": ("Four cards labeled A, B, C, and D are placed in a row from left "
                "to right.\nClues:\n"
                "1. A is somewhere to the left of C.\n"
                "2. B is immediately to the right of D.\n"
                "3. C is not at either end.\n"
                "4. A is not adjacent to D.\n"
                "What is the order from left to right? "
                "Answer as four letters separated by commas."),
     "scoring_type": "constraint", "verifier": "verify_card_order",
     "max_tokens": 512},

    # ── Language Understanding (87-92) ───────────────────────────────────
    {"id": 87, "category": CAT_LANG, "difficulty": "easy",
     "prompt": ("Which meaning of the word \"meticulous\" fits this sentence best?\n"
                "\"The archivist kept meticulous records of every document.\"\n"
                "A) Careful\nB) Careless\nC) Loud\nD) Brief\n"
                "Answer with just the letter."),
     "scoring_type": "exact_match", "expected": ["a"],
     "max_tokens": 64},

    {"id": 88, "category": CAT_LANG, "difficulty": "easy",
     "prompt": ("What is the tone of this sentence?\n"
                "\"Wonderful—another software update right before the presentation.\"\n"
                "Answer with one word: Sarcastic, Grateful, or Neutral."),
     "scoring_type": "exact_match", "expected": ["sarcastic"],
     "max_tokens": 64},

    {"id": 89, "category": CAT_LANG, "difficulty": "medium",
     "prompt": ("What does this sentence presuppose?\n"
                "\"Leila stopped commuting by car last month.\"\n"
                "Answer in one short sentence."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["used to", "previously", "before", "had been", "was commut"], 0.5),
         (["commut", "car", "drive", "drove", "driving"], 0.5),
     ], "max_tokens": 256},

    {"id": 90, "category": CAT_LANG, "difficulty": "medium",
     "prompt": ("Read the two statements:\n"
                "Statement 1: \"Every lamp in the hallway was switched off.\"\n"
                "Statement 2: \"At least one hallway lamp was on.\"\n"
                "Is Statement 2 an Entailment, Contradiction, or Neutral with "
                "respect to Statement 1? Answer with one word."),
     "scoring_type": "exact_match", "expected": ["contradiction"],
     "max_tokens": 64},

    {"id": 91, "category": CAT_LANG, "difficulty": "hard",
     "prompt": ("A hiring manager asks, \"Would Samira be good in a client-facing "
                "role?\"\nYou reply, \"Her reports are always perfectly formatted.\"\n"
                "What are you implying? Answer in one sentence."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["not", "lack", "weak", "doubt", "avoid", "deflect", "faint praise",
           "damning", "indirect", "unsuitable", "doesn't", "without",
           "inadequate", "negative", "insufficient"], 0.5),
         (["client", "interpersonal", "communication", "people", "social",
           "interact", "face", "soft skill"], 0.5),
     ], "max_tokens": 512},

    {"id": 92, "category": CAT_LANG, "difficulty": "hard",
     "prompt": ("In the sentence \"Even Farid passed the qualifying exam,\" what "
                "does the word \"even\" imply about Farid? Answer in one sentence."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["least expected", "unlikely", "surprising", "not expected",
           "unexpected", "least likely", "wasn't expected", "improbable",
           "surprise"], 0.5),
         (["pass", "succeed", "exam", "qualifying", "capable"], 0.5),
     ], "max_tokens": 256},

    # ── Persian Language Understanding (93-98) ───────────────────────────
    {"id": 93, "category": CAT_FA, "difficulty": "easy",
     "prompt": "متضادِ «کوتاه» چیست؟ فقط یک کلمه بنویسید.",
     "scoring_type": "exact_match",
     "expected": ["بلند", "tall", "long", "high"],
     "max_tokens": 64},

    {"id": 94, "category": CAT_FA, "difficulty": "easy",
     "prompt": "جملهٔ «The window is open.» را به فارسی ترجمه کنید.",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["پنجره"], 0.5),
         (["باز"], 0.5),
     ], "max_tokens": 256},

    {"id": 95, "category": CAT_FA, "difficulty": "medium",
     "prompt": ("در جملهٔ «کتابِ جدیدِ معلم روی میز است»، واژهٔ «روی» "
                "چه نقشی دارد؟ فقط نام نقش را بنویسید."),
     "scoring_type": "exact_match",
     "expected": ["حرف اضافه", "حرف\u200cاضافه", "preposition"],
     "max_tokens": 64},

    {"id": 96, "category": CAT_FA, "difficulty": "medium",
     "prompt": "معنیِ اصطلاحِ «آب از سرش گذشته» چیست؟ کوتاه توضیح دهید.",
     "scoring_type": "keyword",
     "keyword_groups": [
         (["دیر", "گذشته", "فایده", "late", "beyond", "past the point",
           "too late", "بی\u200cفایده", "نمی\u200cتوان"], 0.5),
         (["کار", "وضعیت", "اوضاع", "situation", "دیگر",
           "امید", "جبران", "بازگشت"], 0.5),
     ], "max_tokens": 256},

    {"id": 97, "category": CAT_FA, "difficulty": "hard",
     "prompt": ("مفهومِ این بیت چیست؟\n"
                "«تو نیکی می\u200cکن و در دجله انداز / "
                "که ایزد در بیابانت دهد باز»\n"
                "کوتاه توضیح دهید."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["نیکی", "خوبی", "good", "kindness", "کار خیر",
           "احسان", "نیک"], 0.5),
         (["پاداش", "برگرد", "برمی\u200cگرد", "جبران", "reward",
           "return", "باز", "بازگشت", "نتیجه"], 0.5),
     ], "max_tokens": 512},

    {"id": 98, "category": CAT_FA, "difficulty": "hard",
     "prompt": ("در جملهٔ «اگر زودتر خبر می\u200cداد، می\u200cآمدم» "
                "گوینده چه چیزی را بیان می\u200cکند؟ کوتاه توضیح دهید."),
     "scoring_type": "keyword",
     "keyword_groups": [
         (["نیامد", "نیامده", "نرفت", "didn't come", "did not come",
           "نتوانست", "خبر ندا", "خبر نگرفت"], 0.5),
         (["خلاف واقع", "شرطی", "غیرواقعی", "counterfactual",
           "conditional", "hypothetical", "اگر", "فرضی",
           "خبر", "زودتر", "دیر"], 0.5),
     ], "max_tokens": 256},
]

assert len(QUESTIONS) == 98, f"Expected 98 questions, got {len(QUESTIONS)}"

# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════


def clean_response(text: str) -> str:
    """Strip thinking tags and other artifacts from model output."""
    original_len = len(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"/no_think\s*", "", text)
    cleaned = text.strip()
    if len(cleaned) != original_len:
        log.debug("clean_response: stripped %d chars (think tags / artifacts)",
                  original_len - len(cleaned))
    return cleaned


def extract_code(response: str, function_name: str = "") -> str:
    """Extract Python code from a model response."""
    text = clean_response(response)

    # Try markdown code blocks (```python ... ``` or ``` ... ```)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        log.debug("extract_code: found %d markdown code block(s)", len(blocks))
        if function_name:
            for block in blocks:
                if f"def {function_name}" in block:
                    log.debug("extract_code: matched block with 'def %s' (%d chars)",
                              function_name, len(block))
                    return block.strip()
        chosen = max(blocks, key=len).strip()
        log.debug("extract_code: using longest block (%d chars)", len(chosen))
        return chosen

    # Fallback: find function definition and capture the indented block
    log.debug("extract_code: no markdown blocks, falling back to def-scan")
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            if not function_name or f"def {function_name}" in line:
                start = i
                break

    if start is not None:
        # Also grab preceding import lines
        pre = [ln for ln in lines[:start] if ln.strip().startswith(("import ", "from "))]
        code_lines = pre + [lines[start]]
        for line in lines[start + 1 :]:
            if line == "" or line[0] in " \t":
                code_lines.append(line)
            else:
                break
        result = "\n".join(code_lines).rstrip()
        log.debug("extract_code: captured def block at line %d (%d lines, %d chars)",
                  start, len(code_lines), len(result))
        return result

    log.warning("extract_code: no code block or def found for '%s', returning raw text",
                function_name)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Service management
# ═══════════════════════════════════════════════════════════════════════════════


def stop_all_services() -> None:
    """Stop all model services."""
    log.info("Stopping all %d model services ...", len(MODELS))
    for model in MODELS:
        svc = model["service"]
        log.debug("Stopping service: %s", svc)
        result = subprocess.run(
            ["sudo", "systemctl", "stop", svc],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.debug("Stop %s returned rc=%d: %s", svc, result.returncode,
                      result.stderr.strip())
    time.sleep(2)
    log.info("All model services stopped")


def start_service(service_name: str) -> None:
    """Start a systemd service."""
    log.info("Starting service: %s", service_name)
    result = subprocess.run(
        ["sudo", "systemctl", "start", service_name],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.error("systemctl start %s failed (rc=%d): %s",
                  service_name, result.returncode, result.stderr.strip())
        raise RuntimeError(f"Failed to start {service_name}: {result.stderr.strip()}")
    log.info("Service %s start command accepted", service_name)


def wait_for_health(timeout: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll /health until the server is ready or timeout."""
    log.info("Polling %s (timeout %ds) ...", HEALTH_URL, timeout)
    start = time.time()
    attempts = 0
    while time.time() - start < timeout:
        attempts += 1
        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if resp.status == 200:
                body = json.loads(resp.read())
                status = body.get("status")
                if status == "ok":
                    elapsed = time.time() - start
                    log.info("Health check passed after %d attempts (%.1fs)",
                             attempts, elapsed)
                    return True
                log.debug("Health response status=%s (attempt %d)", status, attempts)
        except Exception as exc:
            log.debug("Health poll attempt %d: %s", attempts, exc)
        time.sleep(2)
    log.error("Health check timed out after %ds (%d attempts)", timeout, attempts)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Query
# ═══════════════════════════════════════════════════════════════════════════════


def get_vram_usage_mb() -> float | None:
    """Read current GPU memory usage in MB via tegrastats or nvidia-smi."""
    # Try tegrastats-style /sys node (Jetson)
    try:
        with open("/sys/kernel/debug/nvmap/iovmm/maps") as f:
            # Total mapped memory — rough proxy for VRAM on unified-memory Jetson
            pass
    except Exception:
        pass

    # Try nvidia-smi (discrete GPU)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Try tegrastats snapshot
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--stop", "1"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            m = re.search(r"RAM\s+(\d+)/(\d+)MB", result.stdout)
            if m:
                return float(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def warmup_model(client: OpenAI) -> None:
    """Send a short throwaway query to warm up the model (load KV cache, etc.)."""
    log.info("Warmup: sending throwaway query ...")
    try:
        client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Say hello."},
            ],
            max_tokens=16,
        )
        log.info("Warmup complete")
    except Exception as exc:
        log.warning("Warmup query failed (non-fatal): %s", exc)


def query_model(
    client: OpenAI, prompt: str, max_tokens: int = 256, retries: int = 2,
) -> dict[str, Any]:
    """Send a prompt via streaming to the model. Measures TTFT and wall time."""
    log.debug("query_model: max_tokens=%d, prompt_len=%d chars", max_tokens, len(prompt))
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            ttft_ms: float | None = None
            chunks: list[str] = []
            finish_reason = None

            stream = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if ttft_ms is None:
                            ttft_ms = (time.time() - t0) * 1000
                        chunks.append(delta.content)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

            wall_ms = (time.time() - t0) * 1000
            text = "".join(chunks)

            # Try to get usage/timings from the last chunk's model_extra
            timings: dict[str, Any] = {}
            pt = ct = 0
            try:
                extra = getattr(chunk, "model_extra", None) or {}
                timings = extra.get("timings", {})
                usage = extra.get("usage", {})
                pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                ct = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
            except Exception:
                log.debug("query_model: could not extract timings/usage from stream")

            gen_tps = timings.get("predicted_per_second", 0)
            log.debug("query_model: wall=%.0fms, ttft=%.0fms, prompt_tok=%d, "
                      "compl_tok=%d, gen_tok/s=%.1f, finish=%s, resp_len=%d chars",
                      wall_ms, ttft_ms or 0, pt, ct, gen_tps or 0,
                      finish_reason, len(text))

            return {
                "text": text,
                "finish_reason": finish_reason or "unknown",
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "timings": timings,
                "wall_time_ms": wall_ms,
                "ttft_ms": ttft_ms,
                "error": None,
            }
        except Exception as exc:
            wall_ms = (time.time() - t0) * 1000
            if attempt < retries:
                log.warning("Query attempt %d/%d failed (%.0fms): %s — retrying in 3s",
                            attempt + 1, retries + 1, wall_ms, exc)
                time.sleep(3)
            else:
                log.error("Query failed after %d attempts (%.0fms): %s",
                          retries + 1, wall_ms, exc)
                return {
                    "text": "",
                    "finish_reason": "error",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "timings": {},
                    "wall_time_ms": wall_ms,
                    "ttft_ms": None,
                    "error": str(exc),
                }
    # unreachable
    return {"text": "", "finish_reason": "error", "prompt_tokens": 0,
            "completion_tokens": 0, "timings": {}, "wall_time_ms": 0,
            "ttft_ms": None, "error": "unreachable"}


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════════════════════


def _match_mc_letter(response: str, letter: str) -> bool:
    """Check if a single-letter MC answer is present in the response."""
    resp = response.strip()
    # Short response that IS just the letter
    if resp == letter:
        return True
    # "A)" or "A." at start
    if re.search(rf"(?:^|\n)\s*{letter}\s*[\).\]]", resp):
        return True
    # "answer is A" / "answer: A"
    if re.search(rf"answer\s*(?:is|:)\s*[\(]?{letter}[\)]?\b", resp):
        return True
    # "(A)"
    if re.search(rf"\({letter}\)", resp):
        return True
    # "option A"
    if re.search(rf"option\s+{letter}\b", resp):
        return True
    return False


def score_exact_match(response: str, expected: list[str]) -> float:
    resp = response.lower().strip()
    resp_clean = resp.replace(",", "").replace("\u200c", " ")
    # Normalize Arabic→Persian chars
    resp_fa = resp_clean.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")

    log.debug("score_exact_match: resp_clean='%s' (first 80 chars), %d expected values",
              resp_clean[:80], len(expected))

    for exp in expected:
        exp_l = exp.lower().strip()
        exp_clean = exp_l.replace(",", "").replace("\u200c", " ")
        exp_fa = exp_clean.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")

        # Direct equality
        if resp_clean == exp_clean or resp_fa == exp_fa:
            log.debug("score_exact_match: direct equality match on '%s'", exp)
            return 1.0

        # Non-ASCII (Persian/Arabic) — substring match
        if any(ord(c) > 127 for c in exp):
            if exp_fa in resp_fa:
                log.debug("score_exact_match: Persian substring match on '%s'", exp)
                return 1.0
            continue

        # Single-letter MC
        if len(exp) == 1 and exp.isalpha():
            if _match_mc_letter(resp, exp_l):
                log.debug("score_exact_match: MC letter match on '%s'", exp)
                return 1.0
            continue

        # Word boundary match
        try:
            if re.search(r"\b" + re.escape(exp_clean) + r"\b", resp_clean):
                log.debug("score_exact_match: word-boundary match on '%s'", exp)
                return 1.0
        except re.error:
            if exp_clean in resp_clean:
                log.debug("score_exact_match: fallback substring match on '%s'", exp)
                return 1.0

    log.debug("score_exact_match: no match found")
    return 0.0


def score_keyword(response: str, keyword_groups: list[tuple[list[str], float]]) -> float:
    resp = response.lower()
    # Normalize Persian chars in response for matching
    resp = resp.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    total = 0.0
    hits: list[str] = []
    misses: list[str] = []
    for alternatives, weight in keyword_groups:
        found = False
        for alt in alternatives:
            if alt.lower() in resp:
                total += weight
                hits.append(alt)
                found = True
                break
        if not found:
            misses.append(alternatives[0] if alternatives else "?")
    score = min(total, 1.0)
    log.debug("score_keyword: %.2f — hits=%s, misses=%s", score, hits, misses)
    return score


def score_code_exec(
    response: str, test_code: str, num_tests: int, function_name: str = "",
) -> float:
    code = extract_code(response, function_name)
    if not code.strip():
        log.warning("score_code_exec(%s): no code extracted from response", function_name)
        return 0.0

    log.debug("score_code_exec(%s): extracted %d chars of code", function_name, len(code))
    full_code = code + "\n\n" + test_code
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write(full_code)
            tmp = f.name

        log.debug("score_code_exec(%s): running %s (timeout=%ds)",
                  function_name, tmp, CODE_EXEC_TIMEOUT_S)
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True,
            timeout=CODE_EXEC_TIMEOUT_S,
        )
        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            log.debug("score_code_exec(%s): process exited rc=%d, stderr: %s",
                      function_name, result.returncode, result.stderr.strip()[:200])

        # Parse RESULT line
        m = re.search(r"RESULT:\s*(\d+)/(\d+)", output)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            score = passed / total
            log.debug("score_code_exec(%s): %d/%d tests passed (%.2f)",
                      function_name, passed, total, score)
            return score

        # Fallback: count PASS lines
        passes = len(re.findall(r"^PASS\s+\d+", output, re.MULTILINE))
        score = passes / num_tests if num_tests > 0 else 0.0
        log.debug("score_code_exec(%s): fallback count — %d/%d PASS lines (%.2f)",
                  function_name, passes, num_tests, score)
        return score

    except subprocess.TimeoutExpired:
        log.warning("score_code_exec(%s): execution timed out after %ds",
                    function_name, CODE_EXEC_TIMEOUT_S)
        return 0.0
    except Exception as exc:
        log.warning("score_code_exec(%s): execution error: %s", function_name, exc)
        return 0.0
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint verifiers
# ═══════════════════════════════════════════════════════════════════════════════



def verify_pet_puzzle(response: str) -> float:
    """Expected: Alice=fish, Bob=cat, Charlie=dog.  1/3 per correct.

    Uses the LAST assignment found for each person so that self-corrections
    in reasoning are respected and early wrong guesses don't get credit.
    """
    expected = {"alice": "fish", "bob": "cat", "charlie": "dog"}
    all_pets = {"cat", "dog", "fish"}
    resp = response.lower()
    lines = resp.split("\n")

    # For each person, find what pet they are LAST assigned to
    last_assignment: dict[str, str] = {}
    for line in lines:
        for person in expected:
            if person not in line:
                continue
            if re.search(r"\bnot\b|doesn't|don't|isn't", line):
                continue
            # Try structured patterns first (most reliable)
            for pet in all_pets:
                if re.search(rf"{person}\s*[=:→\-]+\s*(?:the\s+)?{pet}", line):
                    last_assignment[person] = pet
                    break
                if re.search(rf"{person}\s+(?:has|owns?|gets?|keeps?)\s+(?:a\s+|the\s+)?{pet}", line):
                    last_assignment[person] = pet
                    break
            else:
                # Fallback: co-occurrence — only if a single pet appears near person
                pets_on_line = [p for p in all_pets if p in line]
                if len(pets_on_line) == 1:
                    last_assignment[person] = pets_on_line[0]

    score = 0.0
    matched_pairs: list[str] = []
    for person, expected_pet in expected.items():
        if last_assignment.get(person) == expected_pet:
            score += 1 / 3
            matched_pairs.append(f"{person}={expected_pet}")

    result = min(round(score, 4), 1.0)
    log.debug("verify_pet_puzzle: last_assignment=%s, matched=%s → %.2f",
              last_assignment, matched_pairs, result)
    return result


def verify_five_houses(response: str) -> float:
    """Expected: Red, Blue, Green, White, Yellow.  0.2 per correct position."""
    expected = ["red", "blue", "green", "white", "yellow"]
    resp = response.lower()
    colors: list[str] = []

    # Pattern 1: numbered list  "1: Red" / "Position 1 - Red"
    for i in range(1, 6):
        m = re.search(
            rf"(?:position\s*|house\s*)?{i}\s*[:.)\-=]+\s*(red|blue|green|white|yellow)",
            resp,
        )
        if m:
            colors.append(m.group(1))
    if len(colors) != 5:
        colors = []

    # Pattern 2: comma-separated list
    if not colors:
        m = re.search(
            r"(red|blue|green|yellow|white)"
            r"\s*[,\s]+\s*(red|blue|green|yellow|white)"
            r"\s*[,\s]+\s*(red|blue|green|yellow|white)"
            r"\s*[,\s]+\s*(red|blue|green|yellow|white)"
            r"\s*[,\s]+\s*(red|blue|green|yellow|white)",
            resp,
        )
        if m:
            colors = [m.group(i) for i in range(1, 6)]

    # Fallback: order of first appearance
    if len(colors) != 5:
        found = []
        for color in ["red", "blue", "green", "white", "yellow"]:
            pos = resp.find(color)
            if pos >= 0:
                found.append((pos, color))
        found.sort()
        if len(found) == 5:
            colors = [c for _, c in found]

    if len(colors) != 5:
        log.debug("verify_five_houses: could not extract 5 colors from response")
        return 0.0

    correct = sum(1 for a, b in zip(colors, expected) if a == b)
    score = correct / 5
    log.debug("verify_five_houses: extracted=%s, expected=%s, correct=%d/5 → %.2f",
              colors, expected, correct, score)
    return score


def verify_badge_puzzle(response: str) -> float:
    """Expected: Nadia=green, Omid=blue, Parisa=red.  1/3 per correct."""
    expected = {"nadia": "green", "omid": "blue", "parisa": "red"}
    all_colors = {"red", "blue", "green"}
    resp = response.lower()
    lines = resp.split("\n")

    last_assignment: dict[str, str] = {}
    for line in lines:
        for person in expected:
            if person not in line:
                continue
            if re.search(r"\bnot\b|doesn't|don't|isn't", line):
                continue
            for color in all_colors:
                if re.search(rf"{person}\s*[=:→\-]+\s*(?:the\s+)?{color}", line):
                    last_assignment[person] = color
                    break
                if re.search(rf"{person}\s+(?:has|gets?|receives?|holds?)\s+(?:a\s+|the\s+)?{color}", line):
                    last_assignment[person] = color
                    break
            else:
                colors_on_line = [c for c in all_colors if c in line]
                if len(colors_on_line) == 1:
                    last_assignment[person] = colors_on_line[0]

    score = sum(1 / 3 for p, c in expected.items() if last_assignment.get(p) == c)
    result = min(round(score, 4), 1.0)
    log.debug("verify_badge_puzzle: last_assignment=%s → %.2f", last_assignment, result)
    return result


def verify_race_order(response: str) -> float:
    """Expected: Ari, Darya, Cyrus, Bina.  0.25 per correct position."""
    expected = ["ari", "darya", "cyrus", "bina"]
    resp = response.lower()
    names: list[str] = []

    # Pattern 1: numbered list
    for i in range(1, 5):
        m = re.search(
            rf"(?:position\s*|place\s*)?{i}(?:st|nd|rd|th)?\s*[:.)\-=]+\s*(ari|bina|cyrus|darya)",
            resp,
        )
        if m:
            names.append(m.group(1))
    if len(names) != 4:
        names = []

    # Pattern 2: comma-separated
    if not names:
        m = re.search(
            r"(ari|bina|cyrus|darya)\s*[,\s]+\s*(ari|bina|cyrus|darya)"
            r"\s*[,\s]+\s*(ari|bina|cyrus|darya)\s*[,\s]+\s*(ari|bina|cyrus|darya)",
            resp,
        )
        if m:
            names = [m.group(i) for i in range(1, 5)]

    # Fallback: order of first appearance
    if len(names) != 4:
        found = []
        for name in expected:
            pos = resp.find(name)
            if pos >= 0:
                found.append((pos, name))
        found.sort()
        if len(found) == 4:
            names = [n for _, n in found]

    if len(names) != 4:
        log.debug("verify_race_order: could not extract 4 names from response")
        return 0.0

    correct = sum(1 for a, b in zip(names, expected) if a == b)
    score = correct / 4
    log.debug("verify_race_order: extracted=%s, expected=%s → %.2f", names, expected, score)
    return score


def verify_card_order(response: str) -> float:
    """Expected: A, C, D, B.  0.25 per correct position."""
    expected = ["a", "c", "d", "b"]
    resp = response.lower()
    letters: list[str] = []

    # Pattern 1: comma-separated A, C, D, B
    m = re.search(
        r"\b([abcd])\s*[,\s]+\s*([abcd])\s*[,\s]+\s*([abcd])\s*[,\s]+\s*([abcd])\b",
        resp,
    )
    if m:
        letters = [m.group(i) for i in range(1, 5)]

    # Fallback: order of first solo appearance (skip mentions in clue text)
    if len(letters) != 4:
        # Look after "order" or "answer" keyword if present
        answer_start = 0
        for kw in ["order", "answer", "left to right", "result"]:
            idx = resp.rfind(kw)
            if idx >= 0:
                answer_start = max(answer_start, idx)
        tail = resp[answer_start:]
        found = []
        for letter in ["a", "b", "c", "d"]:
            pos = tail.find(letter)
            if pos >= 0:
                found.append((pos, letter))
        found.sort()
        if len(found) == 4:
            letters = [l for _, l in found]

    if len(letters) != 4:
        log.debug("verify_card_order: could not extract 4 letters from response")
        return 0.0

    correct = sum(1 for a, b in zip(letters, expected) if a == b)
    score = correct / 4
    log.debug("verify_card_order: extracted=%s, expected=%s → %.2f", letters, expected, score)
    return score


CONSTRAINT_VERIFIERS: dict[str, Any] = {
    "verify_pet_puzzle": verify_pet_puzzle,
    "verify_five_houses": verify_five_houses,
    "verify_badge_puzzle": verify_badge_puzzle,
    "verify_race_order": verify_race_order,
    "verify_card_order": verify_card_order,
}


def score_question(question: dict[str, Any], response: str) -> float:
    """Route to the appropriate scorer for a question."""
    qid = question["id"]
    st = question["scoring_type"]
    text = clean_response(response)

    log.debug("score_question Q%d: type=%s, response_len=%d chars", qid, st, len(text))

    if st == "exact_match":
        score = score_exact_match(text, question["expected"])
    elif st == "keyword":
        score = score_keyword(text, question["keyword_groups"])
    elif st == "code_exec":
        score = score_code_exec(
            text,
            question["test_code"],
            question["num_tests"],
            question.get("function_name", ""),
        )
    elif st == "constraint":
        verifier_name = question["verifier"]
        verifier = CONSTRAINT_VERIFIERS[verifier_name]
        log.debug("score_question Q%d: using constraint verifier '%s'", qid, verifier_name)
        score = verifier(text)
    else:
        log.warning("score_question Q%d: unknown scoring type '%s'", qid, st)
        return 0.0

    log.debug("score_question Q%d: final score=%.4f", qid, score)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════════


def write_model_csv(
    model_alias: str, results: list[dict[str, Any]], timestamp: str,
) -> Path:
    path = Path(f"benchmark_{model_alias}_{timestamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "category", "difficulty", "prompt_preview",
            "response_preview", "score_mean", "score_std", "run_scores",
            "prompt_tokens", "completion_tokens",
            "prompt_ms", "predicted_ms", "prompt_tok_s", "gen_tok_s",
            "ttft_ms", "wall_time_ms", "finish_reason",
            "full_response", "error",
        ])
        for r in results:
            t = r["timings"]
            ttft = r.get("ttft_ms")
            w.writerow([
                r["question_id"],
                r["category"],
                r["difficulty"],
                r["prompt"][:80].replace("\n", " "),
                r["response"][:80].replace("\n", " "),
                f"{r['score']:.4f}",
                f"{r.get('score_std', 0.0):.4f}",
                json.dumps(r.get("run_scores", [r["score"]])),
                r["prompt_tokens"],
                r["completion_tokens"],
                t.get("prompt_ms", ""),
                t.get("predicted_ms", ""),
                t.get("prompt_per_second", ""),
                t.get("predicted_per_second", ""),
                f"{ttft:.0f}" if ttft is not None else "",
                f"{r['wall_time_ms']:.0f}",
                r["finish_reason"],
                r["response"],
                r["error"] or "",
            ])
    log.info("Wrote per-model CSV: %s (%d rows)", path, len(results))
    return path


def write_comparison_csv(
    all_results: dict[str, dict[str, Any]], timestamp: str,
) -> Path:
    path = Path(f"benchmark_comparison_{timestamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "params_b", "quant", "overall_score",
            *CATEGORIES,
            "avg_gen_tok_s", "avg_prompt_tok_s", "avg_ttft_ms",
            "vram_mb", "total_wall_time_s",
        ])
        for model_name, data in all_results.items():
            model = data["model"]
            results = data["results"]
            vram_mb = data.get("vram_mb")

            cat_scores: dict[str, float] = {}
            for cat in CATEGORIES:
                cr = [r for r in results if r["category"] == cat]
                cat_scores[cat] = (
                    sum(r["score"] for r in cr) / len(cr) if cr else 0.0
                )

            overall = (
                sum(r["score"] for r in results) / len(results) if results else 0.0
            )

            gen_speeds = [
                r["timings"].get("predicted_per_second", 0)
                for r in results
                if r["timings"].get("predicted_per_second")
            ]
            prompt_speeds = [
                r["timings"].get("prompt_per_second", 0)
                for r in results
                if r["timings"].get("prompt_per_second")
            ]
            ttft_values = [
                r["ttft_ms"] for r in results
                if r.get("ttft_ms") is not None
            ]
            avg_gen = sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0.0
            avg_prompt = sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else 0.0
            avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
            total_wall = sum(r["wall_time_ms"] for r in results) / 1000

            w.writerow([
                model_name,
                model["params_b"],
                model["quant"],
                f"{overall:.4f}",
                *[f"{cat_scores[cat]:.4f}" for cat in CATEGORIES],
                f"{avg_gen:.1f}",
                f"{avg_prompt:.1f}",
                f"{avg_ttft:.0f}" if ttft_values else "",
                f"{vram_mb:.0f}" if vram_mb is not None else "",
                f"{total_wall:.1f}",
            ])
    log.info("Wrote comparison CSV: %s (%d models)", path, len(all_results))
    return path


def write_raw_json(
    all_results: dict[str, dict[str, Any]], timestamp: str,
) -> Path:
    path = Path(f"benchmark_raw_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    size_kb = path.stat().st_size / 1024
    log.info("Wrote raw JSON: %s (%.1f KB)", path, size_kb)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════


def _print_summary(all_results: dict[str, dict[str, Any]]) -> None:
    hdr = (
        f"{'Model':<20} {'Overall':>12}"
        f" {'GenKnow':>12} {'Math':>12} {'Code':>12} {'History':>12}"
        f" {'Logic':>12} {'Lang':>12} {'Persian':>12}"
    )
    log.info("")
    log.info("=" * len(hdr))
    log.info("BENCHMARK RESULTS (N_RUNS=%d, mean ± std)", N_RUNS)
    log.info("=" * len(hdr))
    log.info(hdr)
    log.info("-" * len(hdr))

    for model_name, data in all_results.items():
        results = data["results"]
        if not results:
            continue
        overall_mean = statistics.mean(r["score"] for r in results)
        overall_std = (
            math.sqrt(statistics.mean(r.get("score_std", 0.0) ** 2 for r in results))
        )
        cats_mean: dict[str, float] = {}
        cats_std: dict[str, float] = {}
        for cat in CATEGORIES:
            cr = [r for r in results if r["category"] == cat]
            if cr:
                cats_mean[cat] = statistics.mean(r["score"] for r in cr)
                cats_std[cat] = math.sqrt(
                    statistics.mean(r.get("score_std", 0.0) ** 2 for r in cr)
                )
            else:
                cats_mean[cat] = 0.0
                cats_std[cat] = 0.0

        def _fmt(m: float, s: float) -> str:
            return f"{m:.1%}±{s:.1%}"

        log.info(
            f"{model_name:<20} {_fmt(overall_mean, overall_std):>12}"
            f" {_fmt(cats_mean[CAT_GK], cats_std[CAT_GK]):>12}"
            f" {_fmt(cats_mean[CAT_MATH], cats_std[CAT_MATH]):>12}"
            f" {_fmt(cats_mean[CAT_CODE], cats_std[CAT_CODE]):>12}"
            f" {_fmt(cats_mean[CAT_HIST], cats_std[CAT_HIST]):>12}"
            f" {_fmt(cats_mean[CAT_LOGIC], cats_std[CAT_LOGIC]):>12}"
            f" {_fmt(cats_mean[CAT_LANG], cats_std[CAT_LANG]):>12}"
            f" {_fmt(cats_mean[CAT_FA], cats_std[CAT_FA]):>12}"
        )
    log.info("=" * len(hdr))


def run_benchmark(model_filter: list[str] | None = None) -> None:
    models = MODELS
    if model_filter:
        models = [m for m in MODELS if m["alias"] in model_filter]
        if not models:
            log.error("No models matched filter: %s", model_filter)
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: dict[str, dict[str, Any]] = {}
    total_questions = len(QUESTIONS)

    log.info("LLM Benchmark — %d questions, %d models", total_questions, len(models))
    log.info("Timestamp: %s", timestamp)

    try:
        for mi, model in enumerate(models):
            log.info("")
            log.info("=" * 60)
            log.info("[%d/%d] Model: %s (%s, %s)",
                     mi + 1, len(models), model["name"], model["params_b"], model["quant"])
            log.info("=" * 60)

            stop_all_services()
            try:
                start_service(model["service"])
            except RuntimeError as exc:
                log.error("Skipping %s — %s", model["name"], exc)
                continue
            if not wait_for_health():
                log.error("Skipping %s — health check failed", model["name"])
                continue

            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            warmup_model(client)

            vram_mb = get_vram_usage_mb()
            if vram_mb is not None:
                log.info("VRAM usage after load: %.0f MB", vram_mb)

            results: list[dict[str, Any]] = []

            for qi, question in enumerate(QUESTIONS):
                qid = question["id"]
                log.info(
                    "  [%d/%d] Q%d — %s (%s)",
                    qi + 1, total_questions, qid,
                    question["category"], question["difficulty"],
                )

                run_scores: list[float] = []
                run_results: list[dict[str, Any]] = []
                for run_i in range(N_RUNS):
                    resp = query_model(client, question["prompt"], question["max_tokens"])

                    if resp["error"]:
                        sc = 0.0
                        log.warning("    Run %d error: %s", run_i + 1, resp["error"])
                    else:
                        sc = score_question(question, resp["text"])
                        if resp["finish_reason"] == "length":
                            log.warning("    Run %d truncated (max_tokens)", run_i + 1)

                    run_scores.append(sc)
                    run_results.append(resp)

                    gen_tps = resp["timings"].get("predicted_per_second", "?")
                    log.info(
                        "    Run %d/%d: %.2f | Tokens: %d | Wall: %.0fms | Gen: %s tok/s",
                        run_i + 1, N_RUNS, sc, resp["completion_tokens"],
                        resp["wall_time_ms"], gen_tps,
                    )

                mean_score = statistics.mean(run_scores)
                std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
                # Use the best run's response for the CSV (representative)
                best_idx = run_scores.index(max(run_scores))
                best_resp = run_results[best_idx]

                ttft_values = [r["ttft_ms"] for r in run_results if r.get("ttft_ms") is not None]
                avg_ttft = statistics.mean(ttft_values) if ttft_values else None

                result = {
                    "question_id": qid,
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "prompt": question["prompt"],
                    "response": best_resp["text"],
                    "score": mean_score,
                    "score_std": std_score,
                    "run_scores": run_scores,
                    "prompt_tokens": best_resp["prompt_tokens"],
                    "completion_tokens": best_resp["completion_tokens"],
                    "timings": best_resp["timings"],
                    "wall_time_ms": statistics.mean(
                        r["wall_time_ms"] for r in run_results
                    ),
                    "ttft_ms": avg_ttft,
                    "finish_reason": best_resp["finish_reason"],
                    "error": best_resp["error"],
                }
                results.append(result)

                log.info(
                    "    Mean: %.2f ± %.2f (%s)",
                    mean_score, std_score,
                    [f"{s:.2f}" for s in run_scores],
                )

            all_results[model["name"]] = {
                "model": model, "results": results, "vram_mb": vram_mb,
            }
            write_model_csv(model["alias"], results, timestamp)

            # Per-model summary
            overall = sum(r["score"] for r in results) / len(results) if results else 0
            log.info("Model %s overall: %.1f%%", model["name"], overall * 100)

    except KeyboardInterrupt:
        log.warning("Benchmark interrupted by user")
    finally:
        stop_all_services()

    if all_results:
        write_comparison_csv(all_results, timestamp)
        write_raw_json(all_results, timestamp)
        _print_summary(all_results)
    else:
        log.error("No results collected")

    log.info("Benchmark complete")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Optional: pass model aliases as CLI args to run only those models
    # e.g.  uv run llm_benchmark.py qwen3.5-2b qwen3.5-0.8b qwen3.5-35b-a3b
    filter_aliases = sys.argv[1:] if len(sys.argv) > 1 else None
    run_benchmark(model_filter=filter_aliases)
