import json
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_LABELS = ["negative", "neutral", "positive"]


class RestarHandler:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.config = None

    def initialize(self, context) -> None:
        system_properties = context.system_properties
        model_dir = system_properties.get("model_dir")
        gpu_id = system_properties.get("gpu_id")
        if gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")

        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_config(self.config)
        state_dict = torch.load(
            f"{model_dir}/model.pt", map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data: List[Dict[str, Any]]) -> List[str]:
        texts: List[str] = []
        for row in data:
            payload = row.get("data") or row.get("body") or row
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if "text" in payload:
                texts.append(payload["text"])
            elif "texts" in payload:
                texts.extend(payload["texts"])
        return texts

    def inference(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().tolist()

    def postprocess(self, inference_output: List[List[float]]) -> List[Dict[str, Any]]:
        id2label = None
        if self.config is not None:
            id2label = getattr(self.config, "id2label", None)
        if isinstance(id2label, dict):
            labels = [id2label[idx] for idx in sorted(id2label.keys())]
        else:
            labels = DEFAULT_LABELS

        responses = []
        for probs in inference_output:
            best_idx = int(torch.tensor(probs).argmax().item())
            label = labels[best_idx] if best_idx < len(labels) else str(best_idx)
            confidence = float(probs[best_idx])
            responses.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "probs": probs,
                }
            )
        return responses


_service = RestarHandler()


def handle(data, context):
    if not _service.model:
        _service.initialize(context)
    if data is None:
        return []
    texts = _service.preprocess(data)
    if not texts:
        return []
    outputs = _service.inference(texts)
    return _service.postprocess(outputs)
