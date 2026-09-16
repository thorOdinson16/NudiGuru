from app.pipelines.scoring import _combine


def test_combine_averages_pipelines():
    syllables = [{"text": "na", "accuracy": 80}, {"text": "mas", "accuracy": 80}, {"text": "te", "accuracy": 80}]
    hubert = [{"text": "na", "accuracy": 60}, {"text": "mas", "accuracy": 60}, {"text": "te", "accuracy": 60}]
    detailed = {
        "dtw_pipeline": {"accuracy": 80, "syllables": syllables},
        "hubert_pipeline": {"accuracy": 60, "syllables": hubert},
    }

    combined = _combine("w01", detailed)

    assert combined["accuracy_score"] == 70
    assert [s["text"] for s in combined["syllables"]] == ["na", "mas", "te"]
    assert all(s["accuracy"] == 70 for s in combined["syllables"])


def test_combine_single_pipeline():
    detailed = {
        "dtw_pipeline": {
            "accuracy": 55,
            "syllables": [{"text": "na", "accuracy": 55}, {"text": "mas", "accuracy": 55}, {"text": "te", "accuracy": 55}],
        }
    }
    combined = _combine("w01", detailed)
    assert combined["accuracy_score"] == 55
