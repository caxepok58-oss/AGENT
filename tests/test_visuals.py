from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shorts_agent.exceptions import ConfigError
from shorts_agent.visuals.factory import build_visual_provider
from shorts_agent.visuals.generated import GeneratedVisualProvider
from shorts_agent.visuals.stock import PexelsVisualProvider, PixabayVisualProvider


class FakeResponse:
    """Stands in for a requests.Response, including the context-manager use in
    _StockProvider._download (a plain call is also fine: __enter__ just returns self)."""

    def __init__(self, *, json_data=None, content: bytes = b"fake-video-bytes"):
        self._json = json_data
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json

    def iter_content(self, chunk_size: int = 1 << 16):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Stand-in for requests.Session: returns or raises pre-scripted results in call order."""

    def __init__(self, *results):
        self._results = list(results)
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, *, params=None, headers=None, timeout=None, stream=None):
        self.calls.append((url, params))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# --- GeneratedVisualProvider -------------------------------------------------


def test_fetch_writes_a_correctly_sized_png(tmp_path):
    provider = GeneratedVisualProvider(width=64, height=128)

    asset = provider.fetch("cats and dogs", tmp_path, scene_index=0)

    assert asset.kind == "image"
    assert asset.source == "generated"
    with Image.open(asset.path) as image:
        assert image.size == (64, 128)


def test_palette_is_deterministic_for_the_same_keyword_and_scene():
    provider = GeneratedVisualProvider()

    assert provider._palette("cats", 0) == provider._palette("cats", 0)


def test_palette_differs_across_scenes_for_the_same_keyword():
    """Successive scenes should look different even for one topic (module docstring)."""
    provider = GeneratedVisualProvider()

    assert provider._palette("cats", 0) != provider._palette("cats", 1)


def test_generated_card_has_a_real_vignette(tmp_path):
    """Regression: an earlier implementation's mask was larger than the frame, so
    every pixel in a horizontal row shared one brightness and there was no visible
    vignette at all. The centre must be brighter than both the corner and a point
    on the *same row* near the edge."""
    provider = GeneratedVisualProvider(width=200, height=200)

    asset = provider.fetch("a lit room", tmp_path, scene_index=0)
    with Image.open(asset.path) as image:
        pixels = np.asarray(image, dtype=np.float64)

    center = pixels[100, 100].sum()
    corner = pixels[2, 2].sum()
    same_row_near_edge = pixels[100, 2].sum()

    assert center > corner
    assert center > same_row_near_edge


def test_fetch_raises_a_clear_error_without_video_deps(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in ("numpy", "PIL"):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    from shorts_agent.exceptions import ProviderError

    with pytest.raises(ProviderError, match="Pillow and numpy"):
        GeneratedVisualProvider().fetch("cats", tmp_path, scene_index=0)


# --- stock providers: fallback paths -----------------------------------------


def test_no_api_key_falls_back_to_a_generated_card(tmp_path):
    provider = PexelsVisualProvider(None, session=FakeSession())

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.source == "generated"


def test_search_network_error_falls_back_to_a_generated_card(tmp_path):
    import requests

    provider = PexelsVisualProvider(
        "key", session=FakeSession(requests.ConnectionError("no route to host"))
    )

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.source == "generated"


def test_no_search_results_falls_back_to_a_generated_card(tmp_path):
    provider = PexelsVisualProvider(
        "key", session=FakeSession(FakeResponse(json_data={"videos": []}))
    )

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.source == "generated"


def test_download_failure_falls_back_to_a_generated_card(tmp_path):
    import requests

    search_response = FakeResponse(
        json_data={
            "videos": [
                {
                    "video_files": [
                        {
                            "link": "https://cdn/x.mp4",
                            "file_type": "video/mp4",
                            "width": 1080,
                            "height": 1920,
                        }
                    ]
                }
            ]
        }
    )
    provider = PexelsVisualProvider(
        "key", session=FakeSession(search_response, requests.Timeout("slow cdn"))
    )

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.source == "generated"


# --- Pexels: candidate scoring ------------------------------------------------


def test_pexels_prefers_portrait_over_a_larger_landscape_file(tmp_path):
    search_response = FakeResponse(
        json_data={
            "videos": [
                {
                    "video_files": [
                        {
                            "link": "https://cdn/landscape-4k.mp4",
                            "file_type": "video/mp4",
                            "width": 3840,
                            "height": 2160,
                        },
                        {
                            "link": "https://cdn/portrait.mp4",
                            "file_type": "video/mp4",
                            "width": 1080,
                            "height": 1920,
                        },
                    ]
                }
            ]
        }
    )
    download_response = FakeResponse(content=b"portrait-bytes")
    session = FakeSession(search_response, download_response)
    provider = PexelsVisualProvider("key", session=session)

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.kind == "video"
    assert asset.source == "pexels"
    assert Path(asset.path).read_bytes() == b"portrait-bytes"
    assert session.calls[0][1]["orientation"] == "portrait"


def test_pexels_ignores_non_mp4_files_and_files_missing_dimensions(tmp_path):
    search_response = FakeResponse(
        json_data={
            "videos": [
                {
                    "video_files": [
                        {
                            "link": "https://cdn/bad.mov",
                            "file_type": "video/quicktime",
                            "width": 1080,
                            "height": 1920,
                        },
                        {"link": "https://cdn/no-dims.mp4", "file_type": "video/mp4"},
                        {
                            "link": "https://cdn/good.mp4",
                            "file_type": "video/mp4",
                            "width": 1080,
                            "height": 1920,
                        },
                    ]
                }
            ]
        }
    )
    download_response = FakeResponse(content=b"good-bytes")
    provider = PexelsVisualProvider("key", session=FakeSession(search_response, download_response))

    asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.kind == "video"
    assert Path(asset.path).read_bytes() == b"good-bytes"


# --- Pixabay -------------------------------------------------------------------


def test_pixabay_parses_the_videos_dict_and_picks_the_best_variant(tmp_path):
    search_response = FakeResponse(
        json_data={
            "hits": [
                {
                    "videos": {
                        "tiny": {"url": "https://cdn/tiny.mp4", "width": 240, "height": 426},
                        "large": {"url": "https://cdn/large.mp4", "width": 1080, "height": 1920},
                    }
                }
            ]
        }
    )
    download_response = FakeResponse(content=b"pixabay-bytes")
    provider = PixabayVisualProvider("key", session=FakeSession(search_response, download_response))

    asset = provider.fetch("dogs", tmp_path, scene_index=1)

    assert asset.kind == "video"
    assert asset.source == "pixabay"
    assert Path(asset.path).read_bytes() == b"pixabay-bytes"


def test_pixabay_falls_back_when_hits_have_no_videos_dict(tmp_path):
    search_response = FakeResponse(json_data={"hits": [{"id": 1}]})
    provider = PixabayVisualProvider("key", session=FakeSession(search_response))

    asset = provider.fetch("dogs", tmp_path, scene_index=0)

    assert asset.source == "generated"


# --- factory -------------------------------------------------------------------


def test_factory_builds_the_generated_provider_by_default(config):
    provider = build_visual_provider(config)

    assert provider.name == "generated"


def test_factory_rejects_an_unknown_provider(config):
    config.providers.visuals = "made_up"  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="Unknown visuals provider"):
        build_visual_provider(config)
