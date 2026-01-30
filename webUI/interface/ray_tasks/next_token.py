from typing import List, Dict
import os
import numpy as np
import ray
import torch

class HuggingFacePredictor:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if os.path.exists(model_name):
            self.model = AutoModelForCausalLM.from_pretrained(model_name, from_flax=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval()  # ✅ important

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        texts = list(batch["data"])

        with torch.no_grad():
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            out = self.model(**toks)
            logits = out.logits

            next_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()

            next_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in next_ids]
            next_repr   = [repr(t) for t in next_tokens]  # ✅ shows ' ' as "' '", '\n' as "'\\n'"

        batch["token_id"] = next_ids
        batch["output"] = next_tokens
        batch["output_repr"] = next_repr
        return batch


def run_next_token(lines: List[str], model_name: str) -> List[str]:
    ds = ray.data.from_items([{"data": s} for s in lines if s.strip()])

    predictions = ds.map_batches(
        HuggingFacePredictor,
        fn_constructor_args=[model_name],
        batch_size=8,  # ✅ can be larger now; no generation
        num_cpus=4,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=2),
    )

    rows = predictions.take_all()
    # Here r["output"] is already a list of tokens for that row-batch output format
    # With Ray, it can be stored as list; keep the [0] pattern you used.
    #return [r["output"][0] for r in rows]
    out = []
    for r in rows:
        v = r.get("output")
        # Ray can materialize column as a string OR as a 1-element list
        if isinstance(v, list):
            out.append(v[0] if v else "")
        else:
            out.append(v if v is not None else "")
    return out

