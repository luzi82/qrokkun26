import importlib


def test_v3_renderer_imports() -> None:
    assert importlib.import_module("qrokkun_ai.v3.render_v3_demo") is not None
