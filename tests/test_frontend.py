from pathlib import Path


def test_workbench_contains_required_controls_and_local_assets() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    for element_id in (
        "modeUpload",
        "modeRecord",
        "fileInput",
        "audioPlayer",
        "waveform",
        "recordButton",
        "pauseButton",
        "stopButton",
        "analyzeButton",
        "clearButton",
        "dominantEmotion",
        "probabilities",
        "timeline",
    ):
        assert f'id="{element_id}"' in html
    assert "http://" not in html
    assert "https://" not in html
    assert "/static/app.js" in html


def test_styles_define_responsive_layout_and_four_emotion_colors() -> None:
    css = Path("app/static/styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 860px)" in css
    assert "--neutral:" in css
    assert "--happy:" in css
    assert "--anger:" in css
    assert "--sad:" in css
    assert "overflow-x: auto" in css
