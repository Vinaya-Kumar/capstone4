from typing import List, Dict
import numpy as np
import ray
import torch

class EmbeddingPredictor:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        texts = list(batch["data"])
        with torch.no_grad():
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            out = self.model(**tokens)
            last_hidden = out.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / counts).cpu().numpy()
        batch["embedding"] = [e.tolist() for e in emb]
        return batch



def run_embeddings(lines: List[str], model_name: str) -> List[List[float]]:
    ds = ray.data.from_items([{"data": s} for s in lines if s.strip()])

    result = ds.map_batches(
        EmbeddingPredictor,
        fn_constructor_args=[model_name],
        batch_size=8,
        num_cpus=4,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=2),
    )

    rows = result.take_all()

    out = []
    for r in rows:
        v = r.get("embedding")
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], (float, int))):
            out.append(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            out.append(v[0])
        else:
            out.append([])
    return out



    


