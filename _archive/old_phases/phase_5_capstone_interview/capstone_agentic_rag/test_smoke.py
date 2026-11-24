def test_capstone_smoke():
    # First smoke test: ensure the project entry point is present.
    from pathlib import Path
    assert Path(__file__).with_name("main.py").exists()
