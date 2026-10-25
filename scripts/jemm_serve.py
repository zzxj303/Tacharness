"""Serve JEMM over HTTP (POST /v1/systemone) for the rollout policy (gpt_rollout/jemm.py).

Run through scripts/jemm_serve.sh, in .venv-jemm. Same endpoint and prompt contract as
JEMM's own server (python -m jemm.serve); the differences are local weights and the
precision: int8 loads the base with LLM.int8 and keeps the LoRA, lm_head and vision tower
unquantized, so it fits one 46 GB L40S (about 30 GiB); bf16 needs two such cards.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch
from jemm.model import DecisionModel, _decision_config
from jemm.serve import make_handler
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

ROLLOUT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, default=ROLLOUT_ROOT / "models" / "Qwen3.8-27B")
parser.add_argument("--adapter", type=Path, default=ROLLOUT_ROOT / "models" / "JEMM")
parser.add_argument("--precision", choices=("int8", "bf16"), default="int8")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8790)


def load(base: Path, adapter: Path, precision: str) -> DecisionModel:
    if precision == "int8":
        options = {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["lm_head", "model.visual", "mtp"]
            ),
            "device_map": {"": "cuda:0"},
        }
    else:
        options = {"device_map": "auto"}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", **options
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return DecisionModel(
        model,
        AutoTokenizer.from_pretrained(base),
        name=f"{adapter.name}+{base.name}:{precision}",
        processor=AutoProcessor.from_pretrained(base),
        **_decision_config(str(adapter)),
    )


def main() -> None:
    args = parser.parse_args()
    start = time.perf_counter()
    model = load(args.base, args.adapter, args.precision)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    server.daemon_threads = True
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                "model": model.name,
                "load_s": round(time.perf_counter() - start, 1),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
