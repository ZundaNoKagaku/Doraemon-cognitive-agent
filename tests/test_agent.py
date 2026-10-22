"""認知層（第4–8層）の検査。"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.affective_state import AffectiveState  # noqa: E402
from src.agent import run_sim  # noqa: E402
from src.episodic_memory import Episode, EpisodicMemory  # noqa: E402
from src.planner import choose_goal, preference_probe  # noqa: E402
from src.relationship_model import RelParams, steady_state, step_B  # noqa: E402

P = yaml.safe_load(open(ROOT / "configs/agent.yaml"))["planner"]


def test_safety_mask_excludes_risky_action_regardless_of_value():
    rng = np.random.default_rng(0)
    pn = np.array([1.0, 0.0, 0.0]); sv = np.array([100.0, 0, 0])
    for _ in range(50):
        j, _ = choose_goal(pn, sv, np.zeros(3), np.ones(3), np.ones(3), np.zeros(3), np.zeros(3),
                           np.array([1.0, 0, 0]), P, rng)
        assert j != 0


def test_steady_state_matches_iteration():
    for eta, rho, gain in [(0.05, 0.01, 0.3), (0.5, 0.005, 1.0), (0.01, 0.2, 0.1)]:
        p = RelParams(eta=eta, rho=rho)
        Bs, fp = steady_state(p, gain, 0.2)
        B = 0.0
        for _ in range(50000):
            B = step_B(B, gain * B + 0.2, p)
        assert abs(B - Bs) < 1e-4 and abs(fp) < 1


def test_steady_state_depends_only_on_eta_over_rho():
    a, _ = steady_state(RelParams(eta=0.05, rho=0.01), 0.3, 0.2)
    b, _ = steady_state(RelParams(eta=0.5, rho=0.1), 0.3, 0.2)
    assert abs(a - b) < 1e-9


def test_B_bounded_by_Bmax():
    p = RelParams(eta=1.0, rho=0.005, Bmax=2.0)
    B = 0.0
    for _ in range(1000):
        B = step_B(B, 10.0, p)
        assert 0 <= B <= 2.0


def test_memory_erase_by_person():
    M = EpisodicMemory()
    for j in (0, 1):
        M.store(Episode(0, j, 1, True, 1, 0, 1, 1))
    M.erase(0)
    assert [e.person for e in M.episodes] == [1]
    M.erase(None)
    assert M.episodes == []


def test_affect_shock_changes_E_not_B():
    E = AffectiveState(3)
    E.shock(0, worry=3.0)
    assert E.worry[0] == 3.0 and E.worry[1] == 0.0


def test_preference_probe_ignores_affect_and_mission():
    assert abs(preference_probe(np.zeros(3), P) - 0.5) < 1e-12
    assert preference_probe(np.array([1.0, 0, 0]), P) > 0.5


def test_memory_erase_resets_B_only_in_memory_model():
    hf, _ = run_sim(0, "feedback", 60, P, RelParams(), erase_at=40)
    hm, _ = run_sim(0, "memory", 60, P, RelParams(), erase_at=40)
    assert abs(hf["B"][40, 0] - hf["B"][39, 0]) < 0.1
    assert hm["B"][40, 0] < 0.2 * hm["B"][39, 0]


def test_mission_removal_decays_exposure_B():
    h, _ = run_sim(0, "exposure", 300, P, RelParams(), mission_until=100)
    assert h["B"][-1, 0] < h["B"][99, 0]


def test_none_model_has_no_preference_but_learns_people():
    h, _ = run_sim(0, "none", 200, P, RelParams())
    assert np.all(h["B"] == 0)
    assert np.mean(h["brier"][0][-50:]) < np.mean(h["brier"][0][:5]) + 0.05


def test_simulation_reproducible():
    a, _ = run_sim(3, "feedback", 50, P, RelParams())
    b, _ = run_sim(3, "feedback", 50, P, RelParams())
    assert np.allclose(a["B"], b["B"]) and a["help"] == b["help"]
