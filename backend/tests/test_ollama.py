"""
Unit and Integration Tests for Ollama Offline Mistral Adapter
Tests request formatting, mock API interactions, error resilience, and factory integration.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.models import get_vlm_adapter
from app.models.gemini_adapter import ModelProviderUnavailableError
from app.models.ollama_adapter import OllamaVLMAdapter
from models.vlm import VLMBridge


def test_01_ollama_adapter_initialization():
    adapter = OllamaVLMAdapter(
        base_url="http://localhost:11434",
        model="mistral",
        timeout=45.0,
    )
    assert adapter.name == "ollama/mistral"
    assert adapter.base_url == "http://localhost:11434"
    assert adapter.model_name == "mistral"


def test_02_ollama_adapter_generate_success():
    adapter = OllamaVLMAdapter(base_url="http://localhost:11434", model="mistral")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "mistral",
        "response": "The satellite scene displays an active maritime port with multiple cargo berths.",
        "done": True,
        "total_duration": 420000000,
        "eval_count": 28,
        "eval_duration": 310000000,
    }

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        res = adapter.generate_response(
            image_bytes=b"fake_image_bytes",
            mime_type="image/png",
            prompt="Analyze the port facility.",
            metadata={"width": 512, "height": 512, "is_geotiff": True, "crs": "EPSG:32643"},
            system_instruction="Provide remote sensing analysis.",
        )

        assert mock_post.called
        # Check payload
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["model"] == "mistral"
        assert payload["prompt"] == "Analyze the port facility."
        assert "EPSG:32643" in payload["system"]

        # Check response
        assert "active maritime port" in res.text
        assert res.model_name == "ollama/mistral"
        assert res.raw_metadata.get("provider") == "ollama"


def test_03_ollama_adapter_model_not_found_404():
    adapter = OllamaVLMAdapter(base_url="http://localhost:11434", model="mistral")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model 'mistral' not found"

    with patch.object(httpx.Client, "post", return_value=mock_response):
        with pytest.raises(ModelProviderUnavailableError) as exc_info:
            adapter.generate_response(
                image_bytes=b"dummy",
                mime_type="image/png",
                prompt="test prompt",
            )
        assert "ollama pull mistral" in str(exc_info.value)


def test_04_ollama_adapter_connection_error():
    adapter = OllamaVLMAdapter(base_url="http://localhost:11434", model="mistral")

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(ModelProviderUnavailableError) as exc_info:
            adapter.generate_response(
                image_bytes=b"dummy",
                mime_type="image/png",
                prompt="test prompt",
            )
        assert "Ollama service is unreachable" in str(exc_info.value)
        assert "ollama serve" in str(exc_info.value)


def test_05_ollama_adapter_timeout_error():
    adapter = OllamaVLMAdapter(base_url="http://localhost:11434", model="mistral")

    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(ModelProviderUnavailableError) as exc_info:
            adapter.generate_response(
                image_bytes=b"dummy",
                mime_type="image/png",
                prompt="test prompt",
            )
        assert "timed out" in str(exc_info.value).lower()


def test_06_factory_provider_override():
    adapter = get_vlm_adapter(provider_override="ollama")
    assert isinstance(adapter, OllamaVLMAdapter)
    assert adapter.model_name == "mistral"


def test_07_vlm_bridge_with_ollama():
    bridge = VLMBridge()
    mock_adapter = OllamaVLMAdapter(model="mistral")
    bridge._vlm = mock_adapter

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "mistral",
        "response": "Based on the detector findings, 5 aircraft are positioned along the southern tarmac.",
        "done": True,
    }

    with patch.object(httpx.Client, "post", return_value=mock_response):
        ans = bridge.synthesize_answer(
            query="How many aircraft are on the tarmac?",
            task="remote_detection",
            results_summary="5 aircraft detected by YOLOv8n-OBB.",
            image_bytes=b"fake_bytes",
        )
        assert "5 aircraft" in ans
