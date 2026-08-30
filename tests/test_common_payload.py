from __future__ import annotations

import pytest

from common.payload import read_json_object


@pytest.mark.parametrize(
    "source",
    (
        '{"key": 1, "key": 2}',
        '{"outer": {"key": 1, "key": 2}}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
    ),
)
def test_strict_json_reader_rejects_duplicates_and_nonfinite_constants(
    tmp_path, source
) -> None:
    path = tmp_path / "payload.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|non-finite"):
        read_json_object(path)
