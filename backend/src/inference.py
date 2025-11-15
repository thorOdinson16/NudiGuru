import base64
import io
import re
import traceback
from typing import Union

import numpy as np
import pysbd
from aksharamukha.transliterate import process as aksharamukha_xlit
from scipy.io.wavfile import write as scipy_wav_write
from TTS.utils.synthesizer import Synthesizer

from .models.common import Language
from .models.request import TTSRequest
from .models.response import AudioConfig, AudioFile, TTSFailureResponse, TTSResponse
from .postprocessor import PostProcessor
from .utils.paragraph_handler import ParagraphHandler
from .utils.text import TextNormalizer


class TextToSpeechEngine:
    def __init__(
        self,
        models: dict,
        allow_transliteration: bool = False,     # DISABLED - Windows safe
        enable_denoiser: bool = True,
    ):
        self.models = models

        # -------------------------
        # NO transliteration engine
        # -------------------------
        self.xlit_engine = None
        code_mixed_found = False

        # Normal components
        self.text_normalizer = TextNormalizer()
        self.paragraph_handler = ParagraphHandler()
        self.sent_seg = pysbd.Segmenter(language="en", clean=True)

        self.orig_sr = 22050
        self.enable_denoiser = enable_denoiser

        if enable_denoiser:
            from src.postprocessor import Denoiser

            self.target_sr = 16000
            self.denoiser = Denoiser(self.orig_sr, self.target_sr)
        else:
            self.target_sr = self.orig_sr

        self.post_processor = PostProcessor(self.target_sr)

        # -------------------------
        # REMOVE enchant fallback
        # -------------------------
        self.enchant_dicts = {}
        self.enchant_tokenizer = None

    def concatenate_chunks(self, wav: np.ndarray, wav_chunk: np.ndarray):
        if type(wav_chunk) != np.ndarray:
            wav_chunk = np.array(wav_chunk)
        if wav is None:
            return wav_chunk
        return np.concatenate([wav, wav_chunk])

    def infer_from_request(
        self, request: TTSRequest, transliterate_roman_to_native: bool = False
    ) -> TTSResponse:
        config = request.config
        lang = config.language.sourceLanguage
        gender = config.gender

        if lang not in self.models:
            return TTSFailureResponse(status_text="Unsupported language!")

        if lang == "brx" and gender == "male":
            return TTSFailureResponse(
                status_text="Sorry, `male` speaker not supported for this language!"
            )

        output_list = []

        for sentence in request.input:
            raw_audio = self.infer_from_text(
                sentence.source,
                lang,
                gender,
            )
            byte_io = io.BytesIO()
            scipy_wav_write(byte_io, self.target_sr, raw_audio)

            encoded_bytes = base64.b64encode(byte_io.read())
            encoded_string = encoded_bytes.decode()
            speech_response = AudioFile(audioContent=encoded_string)

            output_list.append(speech_response)

        audio_config = AudioConfig(language=Language(sourceLanguage=lang))
        return TTSResponse(audio=output_list, config=audio_config)

    def infer_from_text(
        self,
        input_text: str,
        lang: str,
        speaker_name: str,
        transliterate_roman_to_native: bool = False,
    ) -> np.ndarray:

        # Hinglish fallback safety
        split_lang = lang
        if lang == "en" and lang not in self.models and "en+hi" in self.models:
            lang = "en+hi"
            split_lang = "hi"

        input_text, primary_lang, secondary_lang = self.parse_langs_normalise_text(
            input_text, lang
        )

        # NO transliteration
        xlit_paragraph = input_text

        wav = None
        paragraphs = self.paragraph_handler.split_text(xlit_paragraph, split_lang)

        for paragraph in paragraphs:
            paras = []
            for sent in self.sent_seg.segment(paragraph):
                if sent.strip() and not re.match(r"^[_\W]+$", sent.strip()):
                    paras.append(sent.strip())
            paragraph = " ".join(paras)

            wav_chunk = self.models[lang].tts(
                paragraph, speaker_name=speaker_name, style_wav=""
            )
            wav_chunk = self.postprocess_audio(wav_chunk, primary_lang, speaker_name)
            wav = self.concatenate_chunks(wav, wav_chunk)

        return wav

    def parse_langs_normalise_text(
        self, input_text: str, lang: str
    ) -> Union[str, str, str]:

        if lang == "en" and lang not in self.models and "en+hi" in self.models:
            lang = "en+hi"

        if lang == "en+hi":
            primary_lang, secondary_lang = lang.split("+")
        else:
            primary_lang = lang
            secondary_lang = None

        input_text = self.text_normalizer.normalize_text(input_text, primary_lang)

        # NO spell-check transliteration
        return input_text, primary_lang, secondary_lang

    def postprocess_audio(self, wav_chunk, primary_lang, speaker_name):
        if self.enable_denoiser:
            wav_chunk = self.denoiser.denoise(wav_chunk)

        wav_chunk = self.post_processor.process(wav_chunk, primary_lang, speaker_name)
        return wav_chunk

    # NO enchant / NO transliteration functions
    def transliterate_native_words_using_spell_checker(self, *args, **kwargs):
        return args[0]

    def transliterate_sentence(self, input_text, lang):
        return input_text