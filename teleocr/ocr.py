"""TeleOCR runner. 1.2B model, text-content prompt from the model card."""

import sys
from pathlib import Path

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.vlm_runner import main


def _default_rope(config, device=None, seq_len=None, layer_type=None):
    """transformers 5.17 dropped the 'default' entry TeleOCR's code still looks up."""
    base = getattr(config, "rope_theta", 1000000.0)
    try:
        params = config.rope_parameters
    except Exception:
        params = None
    if isinstance(params, dict) and params.get("rope_theta"):
        base = params["rope_theta"]
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim)
    positions = torch.arange(0, dim, 2, dtype=torch.float, device=device)
    return 1.0 / (base ** (positions / dim)), 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)

import transformers as _tf

_NEEDS_5_17_SHIMS = _tf.__version__.split(".")[0] >= "5"


def _patch_remote_cache_position():
    """TeleOCR indexes cache_position, which transformers 5.17 leaves as None."""
    root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not root.is_dir():
        return
    old_prefill = "if cache_position[0] == 0 or self.model.rope_deltas is None:"
    new_prefill = "if cache_position is None or cache_position[0] == 0 or self.model.rope_deltas is None:"
    old_decode = "if cache_position[0] != 0:"
    new_decode = "if cache_position is not None and cache_position[0] != 0:"
    for path in root.glob("StarDoc*/TeleOCR/*/modeling_naviocr.py"):
        text = path.read_text(encoding="utf-8")
        if old_prefill not in text and old_decode not in text:
            continue
        path.write_text(
            text.replace(old_prefill, new_prefill).replace(old_decode, new_decode),
            encoding="utf-8",
        )
        cache_dir = path.parent / "__pycache__"
        if cache_dir.is_dir():
            for compiled in cache_dir.glob("modeling_naviocr*"):
                compiled.unlink()


if _NEEDS_5_17_SHIMS:
    _patch_remote_cache_position()

    import transformers.modeling_utils as _mu

    _orig_tied = _mu.PreTrainedModel.get_expanded_tied_weights_keys

    def _tied_keys(self, all_submodels=False):
        mapping = getattr(self, "_tied_weights_keys", None)
        if isinstance(mapping, list):
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        return _orig_tied(self, all_submodels=all_submodels)

    _mu.PreTrainedModel.get_expanded_tied_weights_keys = _tied_keys

    import inspect

    import transformers.masking_utils as _masks

    def _accept_input_embeds(fn):
        allowed = set(inspect.signature(fn).parameters)

        def wrapped(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            return fn(*args, **{key: value for key, value in kwargs.items() if key in allowed})

        return wrapped

    _masks.create_causal_mask = _accept_input_embeds(_masks.create_causal_mask)
    _masks.create_sliding_window_causal_mask = _accept_input_embeds(
        _masks.create_sliding_window_causal_mask
    )

if __name__ == "__main__":
    main(
        "StarDoc-AI/TeleOCR",
        "Please output the text content from the image.",
        trust_remote_code=True,
        style="teleocr",
        key_mapping={
            "^visual": "model.visual",
            r"^model(?!\.(language_model|visual))": "model.language_model",
        },
    )
