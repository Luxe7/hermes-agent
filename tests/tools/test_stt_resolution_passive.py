"""STT provider RESOLUTION never installs; transcription still does.

Regression: ``_detect_local_backend`` lazy-installed faster-whisper, and it sits in the provider
resolution chain every status surface uses — ``wake.status`` (fired on every desktop
gateway-open), ``/voice status`` and ``GET /api/audio/voice-config``. Each of those could start a
full dependency rebuild, holding the per-install lock long enough to keep a sibling profile's
backend off its port.
"""

import pytest

from tools import transcription_tools


@pytest.fixture
def no_faster_whisper(monkeypatch):
    """A machine without faster-whisper or a whisper CLI, recording install attempts."""
    attempts: list[int] = []
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False)
    monkeypatch.setattr(transcription_tools, "_has_local_command", lambda: False)
    monkeypatch.setattr(transcription_tools, "_try_lazy_install_stt",
                        lambda: attempts.append(1) or False)
    return transcription_tools, attempts


def test_resolution_reports_unavailable_without_installing(no_faster_whisper):
    tools, attempts = no_faster_whisper
    assert tools._detect_local_backend() is None
    assert tools._resolve_explicit_local() == "none"
    assert attempts == []


@pytest.mark.parametrize("action", ["transcribe", "warm"])
def test_explicit_local_actions_reach_lazy_install(
    no_faster_whisper, monkeypatch, tmp_path, action,
):
    from tools.stt_lease import warm_stt_provider

    tools, attempts = no_faster_whisper
    monkeypatch.setattr(tools, "_load_stt_config", lambda: {"provider": "local"})
    monkeypatch.setattr("tools.tool_backend_helpers.read_selection", lambda _: "local")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF" + bytes(4) + b"WAVE")

    if action == "transcribe":
        result = tools.transcribe_audio(str(audio))
        assert result["success"] is False
    else:
        result = warm_stt_provider()
        assert result["action"] == "error"
    assert attempts == [1]
    assert "faster-whisper not installed" in result["error"]
