import json
import logging
import sqlite3
from datetime import datetime

from cynapse.helpers.database.database_manager import SG, seed
from cynapse.helpers.data_models import Camera, QueryPlan
from cynapse.orchestration import build_graph


def test_chat_refreshes_yesterdays_database(tmp_path, monkeypatch, capsys):
    from cynapse import app as cli

    yesterday = datetime(2026, 10, 17, 12, tzinfo=SG)
    today = datetime(2026, 10, 18, 12, tzinfo=SG)
    path = tmp_path / "frames.db"
    seed(path, yesterday)

    class Clock:
        @staticmethod
        def now(tz):
            return today

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr("sys.argv", ["cynapse", "chat", "--nonai", "--db", str(path)])
    requests = iter(["Show CTE frames today", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(requests))
    cli.main()
    output = capsys.readouterr().out
    response = json.loads(output[output.index("{"):])
    assert response["total"] == 145
    assert response["frames"][0]["datetime"] == "2026-10-18T00:00:00+08:00"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT MAX(datetime) FROM frames").fetchone()[0] == today.isoformat()


def test_checkpoint_repeated_queries_restore_registered_types(tmp_path, caplog):
    now = datetime(2026, 1, 3, 12, tzinfo=SG)
    path = tmp_path / "frames.db"
    seed(path, now)
    graph = build_graph(path)
    config = {"configurable": {"thread_id": "repeat"}}
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            state = graph.invoke({"request": "Show CTE today", "now": now}, config)
            assert isinstance(state["plan"], QueryPlan)
            assert state["plan"].camera == Camera.CTE
            assert state["response"]["total"] == 145
        saved = graph.get_state(config)
        assert isinstance(saved.values["plan"], QueryPlan)
    assert "Deserializing unregistered type" not in caplog.text


def test_ai_chat_uses_configured_extractor_and_closes_it(tmp_path, monkeypatch, capsys):
    from cynapse import app as cli

    path = tmp_path / "frames.db"
    path.touch()
    events = []

    class Extractor:
        def close(self):
            events.append("closed")

    extractor = Extractor()

    def configure():
        events.append("configured")
        return extractor

    def build(db, *, extractor):
        assert db == path
        assert extractor is not None
        events.append("built")
        return object()

    monkeypatch.setattr(cli, "seed", lambda *args: None)
    monkeypatch.setattr(cli, "configured_extractor", configure)
    monkeypatch.setattr(cli, "build_graph", build)
    monkeypatch.setattr("sys.argv", ["cynapse", "chat", "--ai", "--db", str(path)])
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")
    cli.main()
    assert events == ["configured", "built", "closed"]


def test_nonai_chat_does_not_initialize_openai(tmp_path, monkeypatch):
    from cynapse import app as cli

    path = tmp_path / "frames.db"
    path.touch()

    def unexpected_model_call():
        raise AssertionError("The nonai version must not initialize OpenAI")

    def build(db, *, extractor):
        assert extractor is None
        return object()

    monkeypatch.setattr(cli, "seed", lambda *args: None)
    monkeypatch.setattr(cli, "configured_extractor", unexpected_model_call)
    monkeypatch.setattr(cli, "build_graph", build)
    monkeypatch.setattr("sys.argv", ["cynapse", "chat", "--nonai", "--db", str(path)])
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")
    cli.main()


def test_chat_requires_exactly_one_mode(monkeypatch, capsys):
    import pytest
    from cynapse import app as cli

    for flags in ([], ["--ai", "--nonai"], ["--model", "openai:test"]):
        monkeypatch.setattr("sys.argv", ["cynapse", "chat", *flags])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
