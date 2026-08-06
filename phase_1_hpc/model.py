# ============================================================
# model.py — Phase 1 (HPC)
# Wrapper around model loading and generation. Adapted from the
# Phase 0 Colab version: drops the forced device_map={"": 0}
# workaround (that was specifically fixing a T4 CPU-offload bug
# under a tight 16GB VRAM ceiling — Stanage's GPUs should have
# meaningfully more headroom), and reads the model name from CLI
# args rather than a single hardcoded constant, since one script
# needs to serve the entire Qwen2.5 -> Llama-3.x progression.
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


class LegalPromptModel:
    def __init__(self, model_name: str):
        # model_name is now REQUIRED, not optional-with-a-default —
        # forces every caller to be explicit about which model in
        # the progression it's running, rather than silently falling
        # back to a single hardcoded default as Phase 0 did.
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self._loaded = True

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Loaded {self.model_name} — GPU memory allocated: {allocated:.2f} GB")
        else:
            print(f"WARNING: Loaded {self.model_name} but no CUDA device detected.")

    def generate(self, prompt_text: str, max_new_tokens: int = None) -> str:
        if not self._loaded:
            raise RuntimeError("Call .load() before .generate().")

        messages = [{"role": "user", "content": prompt_text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[-1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded = self.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )
        return decoded

    def generate_and_parse(self, prompt_text: str, max_new_tokens: int = None):
        raw_output = self.generate(prompt_text, max_new_tokens=max_new_tokens)
        parsed = config.extract_final_answer(raw_output)
        return raw_output, parsed

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        print(f"Unloaded {self.model_name}, GPU memory freed.")