class GoogleTranslator:
    """Best-effort English translation used only for non-Indic number fallback.

    Fails soft: if the `translators` package or network is unavailable the
    original text is returned unchanged, so TTS still works offline.
    """

    def __init__(self):
        self._translate = None
        self.supported_languages: set[str] = set()
        self.custom_lang_map = {
            "mni": "mni-Mtei",
            "raj": "hi",
        }

        try:
            from translators.server import google, _google

            self._translate = google
            self.supported_languages = set(_google.language_map.get("en", []))
        except Exception:  # noqa: BLE001 - optional dependency / offline
            self._translate = None

    def translate(self, text, from_lang, to_lang):
        if self._translate is None:
            return text

        if from_lang in self.custom_lang_map:
            from_lang = self.custom_lang_map[from_lang]
        elif from_lang not in self.supported_languages:
            return text

        if to_lang in self.custom_lang_map:
            to_lang = self.custom_lang_map[to_lang]
        elif to_lang not in self.supported_languages:
            return text

        try:
            return self._translate(text, from_language=from_lang, to_language=to_lang)
        except Exception:  # noqa: BLE001 - never break TTS on a translation error
            return text

    def __call__(self, **kwargs):
        return self.translate(**kwargs)
