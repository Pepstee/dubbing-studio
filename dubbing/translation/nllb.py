from __future__ import annotations

from dubbing.translation.base import TranslationBackend

DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
_NLLB_CODES = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
}


class NLLBTranslationBackend(TranslationBackend):
    """Local NLLB translation for the personal, non-commercial dogfood build."""

    def __init__(
        self,
        model: str = DEFAULT_NLLB_MODEL,
        *,
        device: str = "auto",
        max_new_tokens: int = 256,
        local_files_only: bool = False,
        model_revision: str | None = None,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.model_revision = model_revision
        self._tokenizer = None
        self._model = None

    @property
    def identity(self) -> str:
        return (
            f"nllb:{self.model}:{self.device}:max_new_tokens={self.max_new_tokens}:"
            f"local={self.local_files_only}:revision={self.model_revision or 'unversioned'}"
        )

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model
        if self.local_files_only:
            from pathlib import Path

            if not Path(self.model).is_dir():
                raise RuntimeError(
                    f"offline NLLB model directory not found: {self.model}"
                )
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB dependencies are not installed. "
                "Install dubbing-studio[translation-local]."
            ) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model,
            local_files_only=self.local_files_only,
        )
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device
        return tokenizer, model

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        if source_language not in _NLLB_CODES or target_language not in _NLLB_CODES:
            raise ValueError("NLLB backend supports en, ko, ro, and ru in this product")
        if source_language == target_language:
            return text
        tokenizer, model = self._load()
        tokenizer.src_lang = _NLLB_CODES[source_language]
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(self._resolved_device) for key, value in encoded.items()}
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(
            _NLLB_CODES[target_language]
        )
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
