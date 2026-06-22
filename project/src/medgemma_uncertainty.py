"""Step 2: Classify report-language uncertainty.

Two scorers are provided:

* `rule_based_classify`: a fast, dependency-free keyword baseline used by
  Step 9 (baseline comparison) and as a fallback if MedGemma is not
  installed / not reachable.
* `MedGemmaUncertaintyScorer`: loads `google/medgemma-4b-it` from
  Hugging Face and calls it with a strict-JSON prompt.

Each function/method writes one JSON line per sample to a JSONL cache so
the pipeline can resume after interruption.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from config import (
    MEDGEMMA_MAX_NEW_TOKENS,
    MEDGEMMA_MODEL_NAME,
    MEDGEMMA_SCORES_JSONL,
    MEDGEMMA_TEMPERATURE,
    RULE_SCORES_JSONL,
    SAMPLES_TWO_READERS_CSV,
)

# ---------------------------------------------------------------------------
# Rule-based baseline (Step 9)
# ---------------------------------------------------------------------------

UNCERTAIN_TERMS: List[str] = [
    "possible",
    "possibly",
    "questionable",
    "may represent",
    "could represent",
    "cannot exclude",
    "can't exclude",
    "cannot be excluded",
    "cannot rule out",
    "suspicious for",
    "suggestive of",
    "subtle",
    "ill-defined",
    "ill defined",
    "equivocal",
    "difficult to exclude",
    "difficult to assess",
    "probable",
    "probably",
    "likely",
    "apparent",
    "appears",
    "may be",
    "might",
    "compatible with",
    "consistent with",
]


def rule_based_classify(sentence: str) -> Dict:
    """Return a dict identical in shape to MedGemma output but produced
    by simple keyword matching."""
    if sentence is None:
        sentence = ""
    s_low = sentence.lower()
    triggers = [term for term in UNCERTAIN_TERMS if term in s_low]
    label = "uncertain" if triggers else "certain"
    return {
        "uncertainty_label": label,
        "confidence": 1.0 if triggers else 0.5,
        "uncertainty_triggers": triggers,
        "reason": (
            f"Matched terms: {triggers}" if triggers
            else "No hedge keywords matched."
        ),
    }


# ---------------------------------------------------------------------------
# MedGemma prompt + scorer
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are classifying uncertainty in a chest X-ray radiology report finding sentence.

Sentence:
\"{sentence}\"

Return ONLY valid JSON with this exact schema:
{{
"uncertainty_label": "certain" or "uncertain",
"confidence": float from 0 to 1,
"uncertainty_triggers": list of strings,
"reason": string
}}

Definitions:
- Use "uncertain" if the sentence suggests uncertainty about whether the finding exists, what it represents, or how clearly it is seen.
- Use "certain" if the sentence describes the finding as present or clearly visualized without hedging.
- Do not mark severity words like "mild" or size words like "small" as uncertain unless they appear with hedge words like "possible", "questionable", "may represent", or "cannot exclude".

Examples:
Sentence: "Possible subtle left lower lobe opacity."
Output:
{{"uncertainty_label": "uncertain", "confidence": 0.95, "uncertainty_triggers": ["possible","subtle"], "reason": "The sentence uses possible and subtle, indicating uncertainty about the presence or visibility of the opacity."}}

Sentence: "Left pleural effusion is present."
Output:
{{"uncertainty_label": "certain", "confidence": 0.95, "uncertainty_triggers": [], "reason": "The sentence states the finding is present without hedging."}}

Now classify the sentence above. Return ONLY the JSON object, with no extra text.
"""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first {...} JSON object out of model output."""
    if not text:
        return None
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find a JSON object in the text
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class MedGemmaUncertaintyScorer:
    """Wrapper around the MedGemma instruction-tuned text model."""

    def __init__(
        self,
        model_name: str = MEDGEMMA_MODEL_NAME,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        # Imported lazily so that the rule-based path works without torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Pick a sensible dtype default
        if torch_dtype is None:
            if device == "cuda":
                dtype = torch.bfloat16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left padding so we can batch-generate continuations cleanly.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Single-sample (kept for compatibility / debugging)
    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        except Exception:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        do_sample = MEDGEMMA_TEMPERATURE > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=MEDGEMMA_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=MEDGEMMA_TEMPERATURE if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def classify(self, sentence: str) -> Dict:
        return self.classify_batch([sentence])[0]

    # ------------------------------------------------------------------
    # Batched generation (much faster on GPU)
    # ------------------------------------------------------------------
    def _build_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            return prompt

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        import torch

        texts = [self._build_chat_text(p) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(self.device)

        do_sample = MEDGEMMA_TEMPERATURE > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=MEDGEMMA_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=MEDGEMMA_TEMPERATURE if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Slice off the prompt portion
        prompt_len = enc["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def classify_batch(self, sentences: List[str]) -> List[Dict]:
        prompts = [PROMPT_TEMPLATE.format(sentence=s) for s in sentences]
        texts = self._generate_batch(prompts)
        results: List[Dict] = []
        retry_idx: List[int] = []
        for i, t in enumerate(texts):
            parsed = _extract_json(t)
            if parsed is None:
                retry_idx.append(i)
                results.append(None)  # placeholder
            else:
                results.append(self._normalize_parsed(parsed, t))
        # Retry once for parse failures
        if retry_idx:
            retry_prompts = [prompts[i] for i in retry_idx]
            retry_texts = self._generate_batch(retry_prompts)
            for j, i in enumerate(retry_idx):
                t = retry_texts[j]
                parsed = _extract_json(t)
                if parsed is None:
                    results[i] = {
                        "uncertainty_label": "parse_failed",
                        "confidence": 0.0,
                        "uncertainty_triggers": [],
                        "reason": "Failed to parse MedGemma output as JSON.",
                        "raw": t,
                    }
                else:
                    results[i] = self._normalize_parsed(parsed, t)
        return results

    @staticmethod
    def _normalize_parsed(parsed: dict, raw_text: str) -> Dict:
        lab = str(parsed.get("uncertainty_label", "")).strip().lower()
        if lab not in {"certain", "uncertain"}:
            return {
                "uncertainty_label": "parse_failed",
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "uncertainty_triggers": parsed.get("uncertainty_triggers", []) or [],
                "reason": parsed.get("reason", "") or "",
                "raw": raw_text,
            }
        return {
            "uncertainty_label": lab,
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "uncertainty_triggers": parsed.get("uncertainty_triggers", []) or [],
            "reason": parsed.get("reason", "") or "",
        }


# ---------------------------------------------------------------------------
# Cached scoring driver
# ---------------------------------------------------------------------------


def _load_cached_sample_ids(cache_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not cache_path.exists():
        return done
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("sample_id")
            if sid:
                done.add(sid)
    return done


def score_dataframe(
    df: pd.DataFrame,
    cache_path: Path,
    scorer: str = "rule",
    medgemma: Optional[MedGemmaUncertaintyScorer] = None,
    progress_every: int = 100,
    batch_size: int = 16,
) -> None:
    """Score every row in df and append results to cache_path JSONL.

    Already-scored sample_ids are skipped to allow resume.  For
    `scorer="medgemma"` we batch `batch_size` sentences per generate()
    call for major speedups on GPU.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_cached_sample_ids(cache_path)
    print(f"[{scorer}] Resuming with {len(done)} cached results in {cache_path}")

    todo = df[~df["sample_id"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print(f"[{scorer}] nothing to do.")
        return

    n_new = 0
    if scorer == "rule":
        with cache_path.open("a", encoding="utf-8") as f:
            for _, row in todo.iterrows():
                out = rule_based_classify(row["sentence"])
                record = {"sample_id": row["sample_id"], "sentence": row["sentence"], **out}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_new += 1
                if n_new % progress_every == 0:
                    print(f"[rule] scored {n_new}/{len(todo)} new samples...")
    elif scorer == "medgemma":
        assert medgemma is not None, "MedGemma scorer not provided."
        bs = max(1, int(batch_size))
        with cache_path.open("a", encoding="utf-8") as f:
            for start in range(0, len(todo), bs):
                batch = todo.iloc[start:start + bs]
                sentences = batch["sentence"].tolist()
                outs = medgemma.classify_batch(sentences)
                for (_, row), out in zip(batch.iterrows(), outs):
                    record = {"sample_id": row["sample_id"], "sentence": row["sentence"], **out}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                n_new += len(batch)
                if (start // bs) % max(1, (progress_every // bs)) == 0:
                    print(f"[medgemma] scored {n_new}/{len(todo)} new samples "
                          f"(batch_size={bs})...")
    else:
        raise ValueError(f"Unknown scorer {scorer}")
    print(f"[{scorer}] done. Added {n_new} new samples. Cache: {cache_path}")


def load_scores(cache_path: Path) -> pd.DataFrame:
    rows = []
    if not cache_path.exists():
        return pd.DataFrame(columns=["sample_id", "uncertainty_label", "confidence"])
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorer", choices=["rule", "medgemma"], default="rule",
        help="Which uncertainty scorer to use.",
    )
    parser.add_argument("--input", type=Path, default=SAMPLES_TWO_READERS_CSV)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override JSONL cache path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows (for smoke tests).")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.scorer == "rule":
        cache_path = args.output or RULE_SCORES_JSONL
        score_dataframe(df, cache_path, scorer="rule")
    else:
        cache_path = args.output or MEDGEMMA_SCORES_JSONL
        print("Loading MedGemma model... (this can take a while)")
        scorer = MedGemmaUncertaintyScorer()
        score_dataframe(df, cache_path, scorer="medgemma", medgemma=scorer)


if __name__ == "__main__":
    main()
