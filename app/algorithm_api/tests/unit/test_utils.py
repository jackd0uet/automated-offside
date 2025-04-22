import numpy as np
import pytest

from utils import convert_to_serializable

# Test for convert to serializable success
def test_convert_to_serializable():
    class Dummy: pass

    result = convert_to_serializable({
        "int": np.int32(1),
        "float": np.float32(2.2),
        "array": np.array([3, 4]),
        "list": [np.float64(5.5), {"x": np.int64(6)}, [np.array([7])]],
        "dict": {"nested": np.array([8, 9])},
        "custom": Dummy()
    })

    assert result["int"] == 1
    assert result["float"] == pytest.approx(2.2)
    assert result["array"] == [3, 4]
    assert result["list"][0] == pytest.approx(5.5)
    assert result["list"][1] == {"x": 6}
    assert result["list"][2] == [[7]]
    assert result["dict"]["nested"] == [8, 9]
    assert isinstance(result["custom"], Dummy)
