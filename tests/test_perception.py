"""知覚層（第2–3層）の検査。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.async_fusion import AsyncFuser, Burst, SensorModel, information_gain  # noqa: E402
from src.world_model import GaussianState  # noqa: E402

H1 = np.array([[1.0]])


def fuser(cc=True, Sb=0.0, pi=0.9):
    s = {m: SensorModel(H1, np.array([[0.25]]), np.array([[Sb]]), pi) for m in ("v", "h", "a")}
    return AsyncFuser(s, np.zeros(1), np.eye(1), np.eye(1) * 1e-3, use_common_cause=cc)


def burst(oid, m, tc, ta, val, n=1, degraded=False):
    return Burst(oid, m, tc, ta, np.full((n, 1), val), degraded)


def test_recursion_posterior_feeds_predict():
    st = GaussianState(np.zeros(1), np.eye(1))
    st.update(np.array([1.0]), H1, np.eye(1))
    m, P = st.mean.copy(), st.cov.copy()
    st.predict(2.0, np.eye(1) * 0.1)
    assert np.allclose(st.mean, m) and np.allclose(st.cov, P + 0.2)


def test_predict_cannot_go_backward():
    st = GaussianState(np.zeros(1), np.eye(1), t=1.0)
    with pytest.raises(ValueError):
        st.predict(0.5, np.eye(1))


def test_each_observation_used_once():
    st = GaussianState(np.zeros(1), np.eye(1))
    st.update(np.array([1.0]), H1, np.eye(1), obs_id="x")
    with pytest.raises(RuntimeError):
        st.update(np.array([1.0]), H1, np.eye(1), obs_id="x")


def test_repeated_estimates_do_not_accumulate():
    f = fuser()
    bs = [burst("a", "v", 0.1, 0.2, 0.5), burst("b", "h", 0.3, 0.4, 0.6)]
    s1, _ = f.estimate(bs, 1.0)
    s2, _ = f.estimate(bs, 1.0)
    assert np.allclose(s1.cov, s2.cov) and len(s1.log) == 2


def test_out_of_sequence_equals_in_order():
    f = fuser()
    inorder = [burst("a", "v", 0.1, 0.2, 0.5), burst("b", "h", 0.3, 0.4, 0.7)]
    late = [burst("a", "v", 0.1, 0.9, 0.5), burst("b", "h", 0.3, 0.4, 0.7)]   # a が遅れて届く
    s1, _ = f.estimate(inorder, 1.0)
    s2, _ = f.estimate(late, 1.0)
    assert np.allclose(s1.mean, s2.mean) and np.allclose(s1.cov, s2.cov)


def test_no_future_leakage():
    f = fuser()
    base = [burst("a", "v", 0.1, 0.2, 0.5)]
    fut1 = base + [burst("b", "h", 0.3, 1.5, 3.0)]
    fut2 = base + [burst("b", "h", 0.3, 1.5, -3.0)]
    s1, _ = f.estimate(fut1, 1.0)
    s2, _ = f.estimate(fut2, 1.0)
    assert np.allclose(s1.mean, s2.mean) and np.allclose(s1.cov, s2.cov)


def test_token_count_information_is_bounded_by_shared_bias():
    f = fuser(cc=False, Sb=0.04)
    vs = []
    for n in (1, 10, 1000):
        st, _ = f.estimate([burst("a", "v", 0.1, 0.2, 0.5, n=n)], 1.0)
        vs.append(st.cov[0, 0])
    # S = 0.25/n + 0.04 → 事後分散の下限 = 1/(1 + 1/0.04) ≈ 0.0385
    assert vs[0] > vs[1] > vs[2] > 1 / (1 + 1 / 0.04) - 1e-3


def test_common_cause_downweights_conflict_but_not_reliability():
    f = fuser(cc=True)
    bs = [burst("v", "v", 0.1, 0.2, 0.0, n=4), burst("a", "a", 0.1, 0.2, 0.05, n=4), burst("h", "h", 0.1, 0.2, 3.0, n=4)]
    _, r = f.estimate(bs, 1.0)
    assert r["h"] < 0.5 < r["v"]
    # 単一の感覚しかないとき r = π（矛盾は検出できない）
    _, r1 = f.estimate([burst("h", "h", 0.1, 0.2, 3.0, n=4)], 1.0)
    assert abs(r1["h"] - 0.9) < 1e-9


def test_degraded_flag_inflates_variance():
    f = fuser(cc=False)
    s_ok, _ = f.estimate([burst("v", "v", 0.1, 0.2, 0.5)], 1.0)
    s_bad, _ = f.estimate([burst("v", "v", 0.1, 0.2, 0.5, degraded=True)], 1.0)
    assert s_bad.cov[0, 0] > s_ok.cov[0, 0]


def test_all_degraded_leads_to_abstention():
    f = fuser(cc=True)
    bs = [burst(m, m, 0.1, 0.2, 0.3, degraded=True) for m in ("v", "h", "a")]
    st, _ = f.estimate(bs, 1.0)
    assert np.sqrt(st.cov[0, 0]) > 0.35        # configs/perception.yaml の abstain_sd


def test_information_gain_nonnegative():
    st = GaussianState(np.zeros(2), np.eye(2))
    assert information_gain(st, np.eye(2), np.eye(2) * 0.5) > 0


def test_calibration_uses_training_objects_only(monkeypatch):
    """較正関数は乱数生成器と物体数だけを受け取り、テスト物体を引数に持たない。
    呼び出し中に生成された物体の ID がすべて訓練用（tr 接頭辞）であることを確認する。"""
    import experiments.sensory_fusion as sf
    seen = []
    orig = sf.make_bursts

    def spy(x, rng, cond="clean", x_other=None, oid="o"):
        seen.append(oid)
        return orig(x, rng, cond, x_other, oid)
    monkeypatch.setattr(sf, "make_bursts", spy)
    sf.calibrate(np.random.default_rng(0), 20)
    assert seen and all(o.startswith("tr") for o in seen)
