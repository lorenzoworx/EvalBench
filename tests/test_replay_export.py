from pathlib import Path

import pytest

from evalbench.replay_export import build_replay


@pytest.mark.asyncio
async def test_static_replay_is_graded_with_real_rules_and_honest_pairing() -> None:
    data = await build_replay(Path(__file__).resolve().parents[1])
    assert data["runs"][0]["accuracy"] == pytest.approx(5 / 6)
    assert data["runs"][1]["accuracy"] == 1
    failed = [item for item in data["results"]["baseline-fixture"] if not item["passed"]]
    assert len(failed) == 1
    assert failed[0]["case_id"] == "smoke.length"
    assert not failed[0]["judgments"][0]["passed"]
    comparison = data["comparisons"][0]
    assert comparison["mcnemar"]["exact_p_value"] == 1
    assert comparison["mcnemar"]["candidate_only_passed"] == 1
    assert not comparison["mcnemar"]["significant"]
    assert "Fictional" in data["provenance"]
