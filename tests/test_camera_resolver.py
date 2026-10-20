from datetime import datetime

import pytest

from cynapse.helpers.resolvers import camera_resolver as cameras
from cynapse.helpers.resolvers.camera_resolver import CameraResolutionError, resolve_camera
from cynapse.helpers.database.database_manager import SG, seed
from cynapse.helpers.data_models import Camera
from cynapse.helpers.data_models import Extraction
from cynapse.orchestration import build_graph


@pytest.mark.parametrize("camera", list(Camera))
def test_exact_names_acronyms_and_consistent_combinations(camera):
    for name in (camera.value, camera.name.lower(), f"{camera.value} ({camera.name})", f"{camera.name} ({camera.value})"):
        assert resolve_camera(name) == camera


@pytest.mark.parametrize("mention,expected", [
    ("Kranji Highway", Camera.KJE), ("  kranji   HIGHWAY  ", Camera.KJE),
    ("Pan Island Expy", Camera.PIE), ("Ayer Rajah Highway", Camera.AYE),
    ("Central Express Way", Camera.CTE), ("Tampines", Camera.TPE),
    ("Kallang–Paya Lebar Expressway", Camera.KPE),
    ("Kallang Paya Lebar Highway", Camera.KPE),
    ("Bukit Timah", Camera.BKE), ("Marina Coastal Highway", Camera.MCE),
    ("Tampines Expresway", Camera.TPE), ("Tampins Expressway", Camera.TPE),
    ("Tamipnes Expressway", Camera.TPE), ("Central Expresway", Camera.CTE),
    ("Centrl Expressway", Camera.CTE), ("Kranj Highway", Camera.KJE),
    ("East Cost Parkway", Camera.ECP), ("Bukit Tima Expressway", Camera.BKE),
    ("Marina Costal Expressway", Camera.MCE),
])
def test_aliases_and_small_typos(mention, expected):
    assert resolve_camera(mention) == expected


@pytest.mark.parametrize("mention", [
    "", "the expressway", "Expressway", "Central", "Highway", "Mars Expressway",
    "CPE", "CT", "TPEE", "CTE and PIE", "Central or Tampines Expressway",
    "not Central Expressway", "all cameras", "CTE; DROP TABLE frames",
    "Central Expressway (PIE)", "Tampines Road", "Pan Island except Tuesday",
    "Expressway Expressway", "Totally Unknown Expressway", "Tampines " + "x" * 100,
])
def test_ambiguous_unknown_and_unsafe_mentions(mention):
    with pytest.raises(CameraResolutionError):
        resolve_camera(mention)


def test_small_score_gap_requires_clarification(monkeypatch):
    # Both candidates are plausible typos: the resolver must not pick the top one.
    monkeypatch.setattr(cameras, "STEMS", {Camera.TPE: "tampines", Camera.PIE: "tampine"})
    with pytest.raises(CameraResolutionError):
        resolve_camera("tampins expressway")


def test_common_suffix_does_not_make_unknown_route_confident():
    with pytest.raises(CameraResolutionError):
        resolve_camera("Moon Expressway")


def extracted(mention):
    return Extraction.model_validate({
        "intent": "query", "reset": False, "continuation": False,
        "camera": {"action": "set", "value": mention},
        "date": {"action": "set", "value": "yesterday"},
        "time": {"action": "keep", "value": None},
        "recurrence": {"action": "keep", "value": None},
        "clarification_question": None,
    })


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "frames.db"
    seed(path, datetime(2026, 1, 3, 12, tzinfo=SG))
    return path


@pytest.mark.parametrize("mention,camera", [("Kranji Highway", Camera.KJE), ("Tampines Expresway", Camera.TPE)])
def test_resolved_camera_returns_matching_records(database, mention, camera):
    graph = build_graph(database, extractor=lambda *args: extracted(mention))
    result = graph.invoke({"request": mention, "now": datetime(2026, 1, 3, 12, tzinfo=SG)}, {"configurable": {"thread_id": "camera"}})
    assert result["response"]["total"] == 288
    assert result["response"]["filters"]["camera"] == camera.value
    assert all(frame["camera_name"] == camera.value for frame in result["response"]["frames"])


def test_ambiguous_followup_does_not_query_or_overwrite_context(database, monkeypatch):
    from cynapse import orchestration as module
    output = iter([extracted("CTE"), extracted("CPE")])
    graph = build_graph(database, extractor=lambda *args: next(output))
    config = {"configurable": {"thread_id": "context"}}
    request = {"request": "fixture", "now": datetime(2026, 1, 3, 12, tzinfo=SG)}
    accepted = graph.invoke(request, config)["plan"]
    def forbidden(*args):
        pytest.fail("Ambiguous camera must not execute SQL")
    monkeypatch.setattr("cynapse.agents.database_reader.query", forbidden)
    state = graph.invoke(request, config)
    assert state["response"]["intent"] == "clarify"
    assert state["plan"] == accepted


@pytest.mark.parametrize("mention", ["CTE.", "CTE!", "\"CTE\"", "Central Expressway."])
def test_sentence_punctuation_in_camera_mention(mention):
    assert resolve_camera(mention) == Camera.CTE
