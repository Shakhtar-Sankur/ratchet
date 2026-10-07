"""GSM8K: the prompt format and the reward (1 if the final number is right, else 0)."""

import re

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"

PROMPT = ("Solve the math problem. Think step by step, then give the final answer "
          "on the last line as '#### <number>'.\n\nProblem: {question}")


def reference_answer(answer_field):
    """GSM8K's answer field ends with '#### <number>'."""
    return _normalize(answer_field.split("####")[-1])


def _normalize(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else v


def extract_answer(text):
    """The model's final answer: the number after the last '####', else in the last
    \\boxed{}, else after the last 'answer is', else the last number in the text."""
    for pattern in (r"####\s*(" + _NUM + ")", r"\\boxed\{\s*(" + _NUM + r")\s*\}",
                    r"answer is[:\s]*(" + _NUM + ")"):
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        if found:
            return _normalize(found[-1])
    found = re.findall(_NUM, text)
    return _normalize(found[-1]) if found else None


def reward(text, reference):
    got = extract_answer(text)
    return 1.0 if got is not None and reference is not None and abs(got - reference) < 1e-6 else 0.0
