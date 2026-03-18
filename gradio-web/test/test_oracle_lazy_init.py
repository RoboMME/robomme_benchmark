from __future__ import annotations

import importlib
import sys
import types


def test_oracle_logic_import_does_not_load_sentence_transformer(monkeypatch):
    calls = []

    class _UnusedSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    fake_module = types.SimpleNamespace(
        SentenceTransformer=_UnusedSentenceTransformer,
        util=types.SimpleNamespace(cos_sim=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    sys.modules.pop("oracle_logic", None)

    oracle_logic = importlib.import_module("oracle_logic")
    oracle_logic = importlib.reload(oracle_logic)

    assert calls == []
    assert oracle_logic._NLP_MODEL is None
    assert oracle_logic._NLP_MODEL_LOAD_ATTEMPTED is False
