from __future__ import annotations

from threading import Event, Thread, current_thread
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.ground_truth_window_scope import _compare_scores_for_replay_window


def test_replay_window_builder_patching_is_thread_safe():
    calls: dict[str, float | str] = {}
    errors: list[Exception] = []
    first_inside_compare = Event()
    second_inside_compare = Event()
    first_returned = Event()

    class Session:
        def ripple(self, index: int) -> RippleEvent:
            assert index == 0
            return RippleEvent(0.0, 20.0, 5.0, 1.0, 2.0, 3.0)

    def original_build_emissions(session, encoding, ripple, *args, **kwargs):
        del session, encoding, args, kwargs
        if isinstance(ripple, RippleEvent):
            calls[current_thread().name] = float(ripple.start)
        else:
            calls[current_thread().name] = f"unscoped:{ripple}"
        return object()

    def original_build_clusterless_mark_emissions(*args, **kwargs):
        del args, kwargs
        return object()

    gt = SimpleNamespace(
        build_emissions=original_build_emissions,
        build_clusterless_mark_emissions=original_build_clusterless_mark_emissions,
    )

    def base_compare(root, scores, *args, **kwargs):
        del root, args, kwargs
        label = str(scores["label"].iloc[0])
        if label == "first":
            first_inside_compare.set()
            # Before the fix, the second call can enter and replace the builders.
            # With serialization it remains blocked until this call returns.
            second_inside_compare.wait(timeout=0.2)
        else:
            second_inside_compare.set()
            assert first_returned.wait(timeout=1.0)
        gt.build_emissions(Session(), "encoding", 0)
        return pd.DataFrame({"ok": [True]})

    def run_decode(label: str, replay_window: RippleEvent) -> None:
        try:
            _compare_scores_for_replay_window(
                gt,
                base_compare,
                "root",
                pd.DataFrame({"label": [label]}),
                replay_window,
                (),
                {},
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            if label == "first":
                first_returned.set()

    first = Thread(
        target=run_decode,
        name="first-window",
        args=("first", RippleEvent(1.0, 2.0, 1.5, np.nan, np.nan, np.nan)),
    )
    second = Thread(
        target=run_decode,
        name="second-window",
        args=("second", RippleEvent(10.0, 11.0, 10.5, np.nan, np.nan, np.nan)),
    )

    first.start()
    assert first_inside_compare.wait(timeout=1.0)
    second.start()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == {"first-window": 1.0, "second-window": 10.0}
    assert gt.build_emissions is original_build_emissions
    assert gt.build_clusterless_mark_emissions is original_build_clusterless_mark_emissions
