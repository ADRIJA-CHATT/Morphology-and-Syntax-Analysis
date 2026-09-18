from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpora import validate_fleurs_schema, validate_ud_schema, _reservoir_utterances
from hf_sources import HuggingFaceSources, HFFileInfo
from locality import estimate_information_locality
from morphology import ud_annotated_features
from corpora import UDSentence
from speech_features import aggregate_speech_features
from decay import fit_decay


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fleurs = root / "fleurs.parquet"
        table = pa.table({
            "id": pa.array([1,2,3,4], type=pa.int32()),
            "num_samples": pa.array([16000, 24000, 32000, 20000], type=pa.int32()),
            "transcription": ["this is a sentence", "another sentence here", "information stays local", "language data example"],
            "raw_transcription": ["this is a sentence", "another sentence here", "information stays local", "language data example"],
            "speaker_id": ["a","b","c","d"],
        })
        pq.write_table(table, fleurs)
        fm = validate_fleurs_schema(fleurs)
        assert fm["required_columns_ok"]
        items, _ = _reservoir_utterances(fleurs, "Test", 4, 3, 100, 0.2, 5.0, np.random.default_rng(1))
        assert len(items) == 4
        speech = aggregate_speech_features([x.text for x in items], [x.duration_s for x in items], "eng")
        assert speech["seconds_per_character"] > 0
        # Build an explicit multi-utterance fixture.  Do not reuse a potentially
        # one-shot iterator here: the production KN estimator needs at least
        # two independent utterances so it can create train and held-out splits.
        base_texts = [x.text for x in items]
        locality_texts = [f"{text} example {rep}" for rep in range(20) for text in base_texts]
        assert len(locality_texts) >= 2
        loc = estimate_information_locality(
            "Test",
            utterance_texts=locality_texts,
            seconds_per_character=speech["seconds_per_character"],
            max_lag=3,
            params={
                "kn_train_chars": 1000,
                "kn_valid_chars": 200,
                "kn_eval_positions": 500,
                "train_fraction": 0.8,
            },
            seed=1,
        )
        assert len(loc.information) == 3

        ud = root / "ud.parquet"
        ud_table = pa.table({
            "text": ["I eat fish", "She reads books"],
            "tokens": [["I","eat","fish"],["She","reads","books"]],
            "upos": [["PRON","VERB","NOUN"],["PRON","VERB","NOUN"]],
            "feats": [["_","Mood=Ind|Tense=Pres","Number=Sing"],["_","Mood=Ind|Tense=Pres","Number=Plur"]],
            "head": [[2,0,2],[2,0,2]],
            "deprel": [["nsubj","root","obj"],["nsubj","root","obj"]],
        })
        pq.write_table(ud_table, ud)
        um = validate_ud_schema(ud)
        assert um["required_columns_ok"]
        sentences = [
            UDSentence("I eat fish", ("I","eat","fish"),("PRON","VERB","NOUN"),("_","Mood=Ind|Tense=Pres","Number=Sing"),(2,0,2),("nsubj","root","obj")),
            UDSentence("She reads books", ("She","reads","books"),("PRON","VERB","NOUN"),("_","Mood=Ind|Tense=Pres","Number=Plur"),(2,0,2),("nsubj","root","obj")),
        ]
        uf = ud_annotated_features(sentences, seed=1)
        assert uf["ud_n_tokens"] == 6
        fake = HuggingFaceSources()
        files = [HFFileInfo("en_us/train-00000-of-00001.parquet", 123), HFFileInfo("en_us/validation-00000-of-00001.parquet", 4)]
        assert len(fake.fleurs_train_files(files, "en_us")) == 1
        print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
