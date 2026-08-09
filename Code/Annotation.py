from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import stanza
except ImportError as e:
    raise ImportError("Missing dependency: stanza. Install with: pip install stanza") from e

try:
    from phonemizer import phonemize
except ImportError:
    phonemize = None

try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except ImportError:
    sudachi_dictionary = None
    sudachi_tokenizer = None

try:
    from konlpy.tag import Okt
except ImportError:
    Okt = None

try:
    from pyopenjtalk import g2p as japanese_g2p
except ImportError:
    japanese_g2p = None

try:
    from g2pk2 import G2p as KoreanG2P
except ImportError:
    try:
        from g2pk import G2p as KoreanG2P
    except ImportError:
        KoreanG2P = None

try:
    from pypinyin import pinyin as zh_pinyin, Style as ZhStyle
except ImportError:
    zh_pinyin = None
    ZhStyle = None


# -----------------------------
# Language normalization
# -----------------------------
LANG_ALIASES: Dict[str, str] = {
    "english": "en",
    "en": "en",
    "german": "de",
    "de": "de",
    "dutch": "nl",
    "nl": "nl",
    "swedish": "sv",
    "sv": "sv",
    "french": "fr",
    "fr": "fr",
    "spanish": "es",
    "es": "es",
    "italian": "it",
    "it": "it",
    "portuguese": "pt",
    "pt": "pt",
    "russian": "ru",
    "ru": "ru",
    "polish": "pl",
    "pl": "pl",
    "czech": "cs",
    "cs": "cs",
    "turkish": "tr",
    "tr": "tr",
    "finnish": "fi",
    "fi": "fi",
    "hungarian": "hu",
    "hu": "hu",
    "indonesian": "id",
    "id": "id",
    "vietnamese": "vi",
    "vi": "vi",
    "mandarin chinese": "zh-hans",
    "mandarin": "zh-hans",
    "chinese": "zh-hans",
    "zh": "zh-hans",
    "zh-cn": "zh-hans",
    "zh-hans": "zh-hans",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "hindi": "hi",
    "hi": "hi",
}

# Phonemizer / eSpeak candidate codes.
# Some systems accept base codes, others want regioned codes.
PHONEMIZER_CODES: Dict[str, List[str]] = {
    "en": ["en-us", "en"],
    "de": ["de-de", "de"],
    "nl": ["nl", "nl-be"],
    "sv": ["sv", "sv-se"],
    "fr": ["fr-fr", "fr"],
    "es": ["es-es", "es"],
    "it": ["it-it", "it"],
    "pt": ["pt-pt", "pt"],
    "ru": ["ru"],
    "pl": ["pl"],
    "cs": ["cs"],
    "tr": ["tr"],
    "fi": ["fi"],
    "hu": ["hu"],
    "id": ["id"],
    "vi": ["vi"],
    "hi": ["hi"],
    "zh-hans": ["zh", "cmn", "zh-cn", "zh-hans"],
}

SENTENCE_END_PUNCT = {".", "!", "?", "。", "！", "？"}
PUNCT_RE = re.compile(r"^\W+$", re.UNICODE)

# Rough IPA vowel nuclei for a generic syllable estimate
VOWEL_GROUP_RE = re.compile(r"[aeiouyɑɒæɛɜɞɪʊʌəɐɔøœɶɤɯɨʉ]+", re.IGNORECASE)
HANGUL_SYLLABLE_RE = re.compile(r"[\uac00-\ud7a3]")
KANA_RE = re.compile(r"[ぁ-ゔァ-ヴー]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Annotation:
    sentence_id: int
    token_id: int
    surface: str
    lemma: str
    upos: str
    feats: str
    phonemes: str
    syllables: int
    start_sec: float
    end_sec: float
    source: str  # stanza | sudachi | konlpy | fallback


def normalize_language(user_lang: str) -> str:
    key = user_lang.strip().lower()
    if key not in LANG_ALIASES:
        raise ValueError(
            f"Unsupported language: {user_lang!r}. "
            f"Supported keys include: {', '.join(sorted(set(LANG_ALIASES.keys()) - {'zh-cn'}))}"
        )
    return LANG_ALIASES[key]


def is_punctuation_token(text: str) -> bool:
    return bool(PUNCT_RE.match(text.strip()))


def is_sentence_end_token(text: str) -> bool:
    return text.strip() in SENTENCE_END_PUNCT


def strip_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def simple_sentence_split(text: str) -> List[str]:
    text = strip_ws(text)
    if not text:
        return []
    # Split after strong sentence-ending punctuation.
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def count_hangul_syllables(text: str) -> int:
    return len(HANGUL_SYLLABLE_RE.findall(text))


def count_kana_mora(text: str) -> int:
    # Rough Japanese mora proxy.
    # Excludes whitespace and punctuation; counts kana characters and long vowel marks.
    chars = KANA_RE.findall(text)
    return len(chars)


def count_han_chars(text: str) -> int:
    return len(HAN_RE.findall(text))


def estimate_syllables_from_phonemes(phonemes: str) -> int:
    cleaned = phonemes.strip().lower()
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_RE.findall(cleaned)
    if groups:
        return len(groups)
    # fallback: count word-like chunks if no vowels were found
    chunks = [c for c in re.split(r"\s+", cleaned) if c]
    return max(1, len(chunks))


@lru_cache(maxsize=16)
def get_stanza_pipeline(lang_code: str):
    """
    Build a stanza pipeline. We try tokenization + MWT + POS + lemma.
    If MWT is not available for a language, we fall back to tokenize + POS + lemma.
    """
    try:
        return stanza.Pipeline(
            lang=lang_code,
            processors="tokenize,mwt,pos,lemma",
            use_gpu=False,
            verbose=False,
        )
    except Exception:
        # Download models if missing, then retry.
        stanza.download(lang_code, verbose=False)
        try:
            return stanza.Pipeline(
                lang=lang_code,
                processors="tokenize,mwt,pos,lemma",
                use_gpu=False,
                verbose=False,
            )
        except Exception:
            return stanza.Pipeline(
                lang=lang_code,
                processors="tokenize,pos,lemma",
                use_gpu=False,
                verbose=False,
            )


@lru_cache(maxsize=8)
def get_japanese_tokenizer():
    if sudachi_dictionary is None or sudachi_tokenizer is None:
        raise ImportError(
            "SudachiPy is required for Japanese. Install with: pip install sudachipy sudachidict_core"
        )
    return sudachi_dictionary.Dictionary().create()


@lru_cache(maxsize=1)
def get_korean_okt():
    if Okt is None:
        raise ImportError(
            "KoNLPy is required for Korean. Install with: pip install konlpy"
        )
    return Okt()


@lru_cache(maxsize=1)
def get_korean_g2p():
    if KoreanG2P is None:
        return None
    return KoreanG2P()


def phonemize_token(token: str, lang_code: str) -> str:
    token = token.strip()
    if not token:
        return ""

    if lang_code == "ja":
        if japanese_g2p is None:
            return ""
        try:
            return strip_ws(japanese_g2p(token))
        except Exception:
            return ""

    if lang_code == "ko":
        g2p = get_korean_g2p()
        if g2p is None:
            return ""
        try:
            return strip_ws(g2p(token))
        except Exception:
            return ""

    if lang_code == "zh-hans":
        # pypinyin is a practical syllable-level proxy for Mandarin.
        if zh_pinyin is None or ZhStyle is None:
            return ""
        try:
            syls = zh_pinyin(
                token,
                style=ZhStyle.TONE3,
                strict=False,
                neutral_tone_with_five=True,
                heteronym=False,
            )
            return " ".join(item[0] for item in syls if item and item[0])
        except Exception:
            return ""

    if phonemize is None:
        return ""

    candidates = PHONEMIZER_CODES.get(lang_code, [lang_code])
    for code in candidates:
        try:
            out = phonemize(
                token,
                language=code,
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                njobs=1,
            )
            return strip_ws(out)
        except Exception:
            continue

    return ""


def estimate_syllables(token: str, phonemes: str, lang_code: str) -> int:
    token = token.strip()
    if not token:
        return 0

    if is_punctuation_token(token):
        return 0

    if lang_code == "zh-hans":
        n = count_han_chars(token)
        return n if n > 0 else max(1, estimate_syllables_from_phonemes(phonemes))

    if lang_code == "ja":
        n = count_kana_mora(token)
        if n > 0:
            return n
        n = count_kana_mora(phonemes)
        return n if n > 0 else 1

    if lang_code == "ko":
        n = count_hangul_syllables(token)
        if n > 0:
            return n
        n = estimate_syllables_from_phonemes(phonemes)
        return n if n > 0 else 1

    n = estimate_syllables_from_phonemes(phonemes)
    return n if n > 0 else 1


def duration_and_pause(token: str, syllables: int, syllables_per_sec: float) -> Tuple[float, float]:
    """
    Returns (token_duration, post_token_pause).
    """
    if is_punctuation_token(token):
        # punctuation itself contributes no pronunciation duration
        # but we add a small pause after it
        if token in SENTENCE_END_PUNCT:
            return 0.0, 0.25
        return 0.0, 0.08

    dur = max(0.08, syllables / syllables_per_sec)
    return dur, 0.04


def annotate_with_stanza(text: str, lang_code: str, syllables_per_sec: float) -> List[Annotation]:
    nlp = get_stanza_pipeline(lang_code)
    doc = nlp(text)

    annotations: List[Annotation] = []
    t = 0.0

    for sent_id, sent in enumerate(doc.sentences, start=1):
        words = list(sent.words)
        for token_id, word in enumerate(words, start=1):
            surface = word.text or ""
            lemma = word.lemma or surface
            upos = word.upos or ""
            feats = word.feats or ""
            phon = phonemize_token(surface, lang_code)
            syll = estimate_syllables(surface, phon, lang_code)
            dur, pause = duration_and_pause(surface, syll, syllables_per_sec)

            start_sec = t
            end_sec = t + dur
            t = end_sec + pause

            annotations.append(
                Annotation(
                    sentence_id=sent_id,
                    token_id=token_id,
                    surface=surface,
                    lemma=lemma,
                    upos=upos,
                    feats=feats,
                    phonemes=phon,
                    syllables=syll,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    source="stanza",
                )
            )

        # extra sentence gap after each sentence
        t += 0.12

    return annotations


def annotate_japanese(text: str, syllables_per_sec: float) -> List[Annotation]:
    tokenizer = get_japanese_tokenizer()
    if sudachi_tokenizer is None:
        raise ImportError("SudachiPy tokenizer import failed.")

    mode = sudachi_tokenizer.Tokenizer.SplitMode.C
    sentences = simple_sentence_split(text)

    annotations: List[Annotation] = []
    t = 0.0

    for sent_id, sent_text in enumerate(sentences, start=1):
        morphemes = tokenizer.tokenize(sent_text, mode)
        for token_id, m in enumerate(morphemes, start=1):
            surface = m.surface()
            lemma = m.dictionary_form()
            upos = "/".join(m.part_of_speech())
            feats = ""
            phon = phonemize_token(surface, "ja")
            syll = estimate_syllables(surface, phon, "ja")
            dur, pause = duration_and_pause(surface, syll, syllables_per_sec)

            start_sec = t
            end_sec = t + dur
            t = end_sec + pause

            annotations.append(
                Annotation(
                    sentence_id=sent_id,
                    token_id=token_id,
                    surface=surface,
                    lemma=lemma,
                    upos=upos,
                    feats=feats,
                    phonemes=phon,
                    syllables=syll,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    source="sudachi",
                )
            )
        t += 0.12

    return annotations


def annotate_korean(text: str, syllables_per_sec: float) -> List[Annotation]:
    okt = get_korean_okt()
    sentences = simple_sentence_split(text)

    annotations: List[Annotation] = []
    t = 0.0

    for sent_id, sent_text in enumerate(sentences, start=1):
        # Okt.pos returns (morph, tag) pairs
        morphs = okt.pos(sent_text, norm=True, stem=True)
        for token_id, (surface, tag) in enumerate(morphs, start=1):
            lemma = surface
            upos = tag
            feats = ""
            phon = phonemize_token(surface, "ko")
            syll = estimate_syllables(surface, phon, "ko")
            dur, pause = duration_and_pause(surface, syll, syllables_per_sec)

            start_sec = t
            end_sec = t + dur
            t = end_sec + pause

            annotations.append(
                Annotation(
                    sentence_id=sent_id,
                    token_id=token_id,
                    surface=surface,
                    lemma=lemma,
                    upos=upos,
                    feats=feats,
                    phonemes=phon,
                    syllables=syll,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    source="konlpy",
                )
            )
        t += 0.12

    return annotations


def annotate_chinese(text: str, syllables_per_sec: float) -> List[Annotation]:
    # Use Stanza for tokenization/lemma/POS if available, and pypinyin as a syllable proxy.
    return annotate_with_stanza(text, "zh-hans", syllables_per_sec)


def annotate_text(text: str, lang_code: str, syllables_per_sec: float) -> List[Annotation]:
    if lang_code == "ja":
        return annotate_japanese(text, syllables_per_sec)
    if lang_code == "ko":
        return annotate_korean(text, syllables_per_sec)
    if lang_code == "zh-hans":
        return annotate_chinese(text, syllables_per_sec)
    return annotate_with_stanza(text, lang_code, syllables_per_sec)


def format_plain_text(annotations: List[Annotation]) -> str:
    lines: List[str] = []
    current_sent = None

    for a in annotations:
        if current_sent != a.sentence_id:
            current_sent = a.sentence_id
            lines.append(f"\n[SENTENCE {current_sent}]")

        feats = a.feats if a.feats else "-"
        phon = a.phonemes if a.phonemes else "-"
        lines.append(
            f"{a.start_sec:8.2f}-{a.end_sec:8.2f}  "
            f"{a.surface}\tlemma={a.lemma}\tupos={a.upos}\tfeats={feats}\t"
            f"phon={phon}\tsyll={a.syllables}\tsrc={a.source}"
        )

    return "\n".join(lines).lstrip("\n") + "\n"


def write_outputs(annotations: List[Annotation], output_base: Path) -> None:
    tsv_path = output_base.with_suffix(".annotated.tsv")
    txt_path = output_base.with_suffix(".annotated.txt")
    json_path = output_base.with_suffix(".annotated.json")

    # TSV
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "sentence_id",
                "token_id",
                "surface",
                "lemma",
                "upos",
                "feats",
                "phonemes",
                "syllables",
                "start_sec",
                "end_sec",
                "source",
            ]
        )
        for a in annotations:
            writer.writerow(
                [
                    a.sentence_id,
                    a.token_id,
                    a.surface,
                    a.lemma,
                    a.upos,
                    a.feats,
                    a.phonemes,
                    a.syllables,
                    f"{a.start_sec:.4f}",
                    f"{a.end_sec:.4f}",
                    a.source,
                ]
            )

    # TXT
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(format_plain_text(annotations))

    # JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in annotations], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create time- and morpheme-annotated text from a .txt file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .txt file",
    )
    parser.add_argument(
        "--language",
        "-l",
        required=True,
        type=str,
        help="Language name or code, e.g. English, German, Hindi, ja, ko, zh",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file base path (without extension). Default: input filename stem",
    )
    parser.add_argument(
        "--syllables-per-second",
        type=float,
        default=4.5,
        help="Estimated syllable rate used for timing approximation (default: 4.5)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    lang_code = normalize_language(args.language)

    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input file is empty.")

    annotations = annotate_text(
        text=text,
        lang_code=lang_code,
        syllables_per_sec=args.syllables_per_second,
    )

    if args.output:
        output_base = Path(args.output)
    else:
        output_base = input_path.with_suffix("")

    write_outputs(annotations, output_base)

    total_time = annotations[-1].end_sec if annotations else 0.0
    print(f"Done.")
    print(f"Wrote: {output_base.with_suffix('.annotated.tsv')}")
    print(f"Wrote: {output_base.with_suffix('.annotated.txt')}")
    print(f"Estimated total duration: {total_time:.2f} sec")


if __name__ == "__main__":
    main()