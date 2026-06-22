"""Step 2: Classify report-language uncertainty.

Two scorers are provided:

* `rule_based_classify`: a fast, dependency-free keyword baseline used by
  Step 9 (baseline comparison) and as a fallback if MedGemma is not
  installed / not reachable.
* `MedGemmaUncertaintyScorer`: loads `google/medgemma-4b-it` from
  Hugging Face and calls it with a strict-JSON prompt.

Each function/method writes one JSON line per sample to a JSONL cache so
the pipeline can resume after interruption.

Implementation note
-------------------
MedGemma 4B-IT is a Gemma 3 *image-text-to-text* model.  Loading it
with `AutoModelForCausalLM` throws `ValueError: Unrecognized
configuration class ... for AutoModelForCausalLM` on `transformers>=4.50`.
The scorer therefore uses `AutoProcessor` + `AutoModelForImageTextToText`
and calls it with text-only chat messages.
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

# Spatial-uncertainty triggers only (boundary / extent / visibility).
# Diagnostic hedges like "possible", "cannot exclude", "suspicious for" are
# intentionally excluded -- this study tests whether *spatial* language in
# the report tracks spatial inter-reader disagreement.
UNCERTAIN_TERMS: List[str] = [
    "ill-defined",
    "ill defined",
    "poorly defined",
    "poorly-defined",
    "indistinct",
    "vague",
    "subtle",
    "faint",
    "hazy",
    "barely visible",
    "blurred",
    "blurry",
    "fuzzy",
    "obscured",
    "ill-circumscribed",
    "ill circumscribed",
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

PROMPT_TEMPLATE = """You are classifying SPATIAL uncertainty in a chest X-ray radiology report finding sentence.

We are asking ONE specific question: does the sentence's language indicate that the
LOCATION, BOUNDARIES, EXTENT, or VISIBILITY of the finding is unclear, ill-defined,
indistinct, faint, or hard to delineate spatially?

We are NOT asking about diagnostic uncertainty (whether the finding exists or what
disease it represents). Diagnostic hedges alone do NOT count as spatial uncertainty.

Sentence:
\"{sentence}\"

Return ONLY valid JSON with this exact schema:
{{
"uncertainty_label": "certain" or "uncertain",
"confidence": float from 0 to 1,
"uncertainty_triggers": list of strings,
"reason": string
}}

Decision rules:
- Label "uncertain" ONLY when the sentence uses language indicating that the
  spatial extent, boundary, location, or visibility of the finding is unclear.
  Typical spatial-uncertainty triggers include:
    "ill-defined", "ill defined", "poorly defined", "ill-circumscribed",
    "indistinct", "vague", "blurred", "fuzzy", "hazy", "obscured",
    "subtle", "faint", "barely visible".
- Label "certain" when the spatial extent / boundary of the finding is described
  as clear, well-defined, well-circumscribed, sharply marginated, or simply named
  without spatial hedging.
- IMPORTANT: do NOT label "uncertain" just because the sentence uses diagnostic
  hedges such as "possible", "possibly", "cannot exclude", "may represent",
  "suspicious for", "suggestive of", "probable", "likely". Those reflect
  diagnostic uncertainty, not spatial uncertainty. A sentence like
  "Possible pneumonia." with no spatial language is "certain" here.
- If a sentence contains BOTH diagnostic and spatial hedging
  (e.g. "possible subtle ill-defined opacity"), it IS "uncertain" because of the
  spatial words "subtle" and "ill-defined" — list only the spatial words in
  uncertainty_triggers.
- Size or severity words ("small", "mild", "large", "extensive") are NOT spatial
  uncertainty and must not be triggers.

Examples:
Sentence: "Ill-defined left lower lobe opacity."
Output:
{{"uncertainty_label": "uncertain", "confidence": 0.95, "uncertainty_triggers": ["ill-defined"], "reason": "The boundaries of the opacity are described as ill-defined."}}

Sentence: "Subtle faint opacity in the left lung base."
Output:
{{"uncertainty_label": "uncertain", "confidence": 0.9, "uncertainty_triggers": ["subtle","faint"], "reason": "Subtle and faint indicate the finding is barely visible / poorly delineated."}}

Sentence: "Possible pneumonia."
Output:
{{"uncertainty_label": "certain", "confidence": 0.85, "uncertainty_triggers": [], "reason": "Only diagnostic hedging ('possible') — no spatial uncertainty language."}}

Sentence: "Well-circumscribed right lower lobe nodule."
Output:
{{"uncertainty_label": "certain", "confidence": 0.95, "uncertainty_triggers": [], "reason": "The finding has well-defined boundaries."}}

Sentence: "Large left pleural effusion is present."
Output:
{{"uncertainty_label": "certain", "confidence": 0.95, "uncertainty_triggers": [], "reason": "The finding is clearly described; 'large' is size, not spatial uncertainty."}}

Now classify the sentence above. Return ONLY the JSON object, with no extra text.
"""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced {...} JSON object out of model output."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first balanced {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = text[start:i + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    return None
    return None


class MedGemmaUncertaintyScorer:
    """Wrapper around the MedGemma 4B-IT instruction-tuned multimodal model."""

    def __init__(
        self,
        model_name: str = MEDGEMMA_MODEL_NAME,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        import torch
        from transformers import AutoProcessor

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # dtype
        if torch_dtype is None:
            if device == "cuda":
                dtype = torch.bfloat16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.dtype = dtype

        # Try the multimodal class first (Gemma 3 / MedGemma 4B is image-text-to-text).
        # Fall back to CausalLM for any text-only Gemma variant.
        model = None
        last_err = None
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=dtype,
            )
        except Exception as e:
            last_err = e
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load MedGemma model {model_name!r}. "
                    f"First error: {last_err!r}. Second error: {e2!r}. "
                    "Ensure (1) you accepted the license at "
                    "https://huggingface.co/google/medgemma-4b-it and "
                    "(2) `transformers>=4.50` is installed."
                ) from e2
        self.model = model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        # Make batched generation deterministic and clean.
        tok = getattr(self.processor, "tokenizer", None) or self.processor
        tok.padding_side = "left"
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str):
        # MedGemma uses Gemma-3 multimodal chat template: each content item
        # is a dict with a "type" field.
        return [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        import torch

        all_messages = [self._build_messages(p) for p in prompts]
        # Render to plain strings via the chat template, then tokenize/pad as a batch.
        chat_texts: List[str] = []
        for msgs in all_messages:
            try:
                chat_texts.append(
                    self.processor.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=False,
                    )
                )
            except Exception:
                # Fall back to plain prompt if chat template unavailable.
                chat_texts.append(prompts[len(chat_texts)])

        tok = self.tokenizer
        enc = tok(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        do_sample = MEDGEMMA_TEMPERATURE > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=MEDGEMMA_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=MEDGEMMA_TEMPERATURE if do_sample else 1.0,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    def classify_batch(self, sentences: List[str]) -> List[Dict]:
        prompts = [PROMPT_TEMPLATE.format(sentence=s) for s in sentences]
        texts = self._generate_batch(prompts)
        results: List[Optional[Dict]] = []
        retry_idx: List[int] = []
        for i, t in enumerate(texts):
            parsed = _extract_json(t)
            if parsed is None:
                retry_idx.append(i)
                results.append(None)
            else:
                results.append(self._normalize_parsed(parsed, t))
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
                        "raw": t[:500],
                    }
                else:
                    results[i] = self._normalize_parsed(parsed, t)
        return results  # type: ignore[return-value]

    def classify(self, sentence: str) -> Dict:
        return self.classify_batch([sentence])[0]

    @staticmethod
    def _normalize_parsed(parsed: dict, raw_text: str) -> Dict:
        lab = str(parsed.get("uncertainty_label", "")).strip().lower()
        if lab not in {"certain", "uncertain"}:
            return {
                "uncertainty_label": "parse_failed",
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "uncertainty_triggers": parsed.get("uncertainty_triggers", []) or [],
                "reason": parsed.get("reason", "") or "",
                "raw": raw_text[:500],
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
    """Score every row in df and append results to cache_path JSONL."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_cached_sample_ids(cache_path)
    print(f"[{scorer}] Resuming with {len(done)} cached results in {cache_path}",
          flush=True)

    todo = df[~df["sample_id"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print(f"[{scorer}] nothing to do.", flush=True)
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
                    print(f"[rule] scored {n_new}/{len(todo)} new samples...",
                          flush=True)
    elif scorer == "medgemma":
        assert medgemma is not None, "MedGemma scorer not provided."
        bs = max(1, int(batch_size))
        with cache_path.open("a", encoding="utf-8") as f:
            for start in range(0, len(todo), bs):
                batch = todo.iloc[start:start + bs]
                sentences = batch["sentence"].tolist()
                outs = medgemma.classify_batch(sentences)
                for (_, row), out in zip(batch.iterrows(), outs):
                    record = {"sample_id": row["sample_id"],
                              "sentence": row["sentence"], **out}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                n_new += len(batch)
                if (start // bs) % max(1, (progress_every // bs)) == 0:
                    print(f"[medgemma] scored {n_new}/{len(todo)} new samples "
                          f"(batch_size={bs})...", flush=True)
    else:
        raise ValueError(f"Unknown scorer {scorer}")
    print(f"[{scorer}] done. Added {n_new} new samples. Cache: {cache_path}",
          flush=True)


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.scorer == "rule":
        cache_path = args.output or RULE_SCORES_JSONL
        score_dataframe(df, cache_path, scorer="rule",
                        batch_size=args.batch_size)
    else:
        cache_path = args.output or MEDGEMMA_SCORES_JSONL
        print("Loading MedGemma model... (this can take a while)", flush=True)
        scorer = MedGemmaUncertaintyScorer(device=args.device)
        score_dataframe(df, cache_path, scorer="medgemma",
                        medgemma=scorer, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
