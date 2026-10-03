from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shorts_agent.config import get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.visuals.factory import build_visual_provider, resolve_visual_provider
from shorts_agent.visuals.generated import GeneratedVisualProvider
from shorts_agent.visuals.local import LocalFootageProvider
from shorts_agent.visuals.stock import (
    PexelsVisualProvider,
    PixabayVisualProvider,
    Rendition,
    best_rendition,
    credit_line,
    first_vertical_else_first,
    search_queries,
)

# The factory reads the cached Settings, which read the real environment and .env.
pytestmark = pytest.mark.usefixtures("clean_settings")


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


def test_pixabay_redacts_its_api_key_from_a_leaked_url_in_logs(tmp_path, caplog):
    """Unlike Pexels (auth header), Pixabay sends its key as a URL query param, and
    a real connection-level exception embeds the full request URL in its message —
    that key must never reach the logs."""
    import requests

    provider = PixabayVisualProvider(
        "super-secret-pixabay-key",
        session=FakeSession(
            requests.ConnectionError(
                "Max retries exceeded with url: /api/videos/?key=super-secret-pixabay-key&q=cats"
            )
        ),
    )

    with caplog.at_level("WARNING"):
        asset = provider.fetch("cats", tmp_path, scene_index=0)

    assert asset.source == "generated"
    assert "super-secret-pixabay-key" not in caplog.text


def test_pixabay_falls_back_when_hits_have_no_videos_dict(tmp_path):
    search_response = FakeResponse(json_data={"hits": [{"id": 1}]})
    provider = PixabayVisualProvider("key", session=FakeSession(search_response))

    asset = provider.fetch("dogs", tmp_path, scene_index=0)

    assert asset.source == "generated"


# --- factory -------------------------------------------------------------------


def with_env(monkeypatch, **variables):
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


def add_footage(config, *names):
    folder = config.resolve_path(config.visuals.footage_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"clip")


def test_with_nothing_set_up_auto_ends_in_generated_cards(config):
    assert config.providers.visuals == "auto"

    assert build_visual_provider(config).name == "generated"


def test_auto_uses_the_footage_folder_once_it_has_clips(config):
    add_footage(config, "beach.mp4")

    assert build_visual_provider(config).name == "local"


def test_auto_ignores_a_footage_folder_with_nothing_usable_in_it(config):
    add_footage(config, "notes.txt", ".DS_Store")

    assert build_visual_provider(config).name == "generated"


def test_auto_prefers_pixabay_when_its_key_is_set(config, monkeypatch):
    with_env(monkeypatch, PIXABAY_API_KEY="k", PEXELS_API_KEY="k")

    assert build_visual_provider(config).name == "pixabay"


def test_auto_uses_pexels_when_that_is_the_only_key(config, monkeypatch):
    with_env(monkeypatch, PEXELS_API_KEY="k")

    assert build_visual_provider(config).name == "pexels"


def test_a_stock_key_wins_over_the_footage_folder_in_auto(config, monkeypatch):
    add_footage(config, "beach.mp4")
    with_env(monkeypatch, PIXABAY_API_KEY="k")

    assert resolve_visual_provider(config, get_settings()) == "pixabay"


@pytest.mark.parametrize("choice", ["local", "generated", "pexels", "pixabay"])
def test_an_explicit_choice_is_never_overridden_by_auto_detection(config, monkeypatch, choice):
    add_footage(config, "beach.mp4")
    with_env(monkeypatch, PIXABAY_API_KEY="k", PEXELS_API_KEY="k")
    config.providers.visuals = choice

    assert build_visual_provider(config).name == choice


def test_stock_scenes_it_cannot_fill_fall_back_to_the_footage_folder(config, monkeypatch):
    add_footage(config, "beach.mp4")
    with_env(monkeypatch, PIXABAY_API_KEY="k")

    provider = build_visual_provider(config)

    assert provider._fallback.name == "local"


def test_stock_scenes_it_cannot_fill_fall_back_to_a_card_without_any_footage(config, monkeypatch):
    with_env(monkeypatch, PIXABAY_API_KEY="k")

    provider = build_visual_provider(config)

    assert provider._fallback.name == "generated"


def test_a_search_that_finds_nothing_uses_the_owners_own_clip(config, tmp_path):
    add_footage(config, "beach.mp4")
    local = LocalFootageProvider(config.resolve_path(config.visuals.footage_dir))
    provider = PexelsVisualProvider(
        "key", session=FakeSession(FakeResponse(json_data={"videos": []})), fallback=local
    )

    asset = provider.fetch("cats", tmp_path / "out", 0)

    assert asset.source == "local"
    assert Path(asset.path).name == "beach.mp4"


def test_a_missing_key_uses_the_injected_fallback_too(tmp_path):
    asset = PexelsVisualProvider(
        None, session=FakeSession(), fallback=GeneratedVisualProvider(width=16, height=16)
    ).fetch("cats", tmp_path, 0)

    assert asset.source == "generated"


def test_the_factory_builds_the_local_provider_on_request(config):
    config.providers.visuals = "local"

    provider = build_visual_provider(config)

    assert isinstance(provider, LocalFootageProvider)
    assert provider.footage_dir == config.resolve_path("config/footage")


def test_factory_rejects_an_unknown_provider(config):
    config.providers.visuals = "made_up"  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="Unknown visuals provider"):
        build_visual_provider(config)


# --- shortening a query that finds nothing --------------------------------------


def test_a_short_keyword_is_searched_as_it_is():
    assert search_queries("coins") == ["coins"]
    assert search_queries("stressed man") == ["stressed man"]


def test_a_long_keyword_is_followed_by_shorter_versions_without_filler_words():
    assert search_queries("person counting cash at kitchen table") == [
        "person counting cash at kitchen table",
        "person counting cash",
        "person counting",
    ]


def test_queries_never_shrink_to_a_single_overly_generic_word():
    assert all(len(q.split()) >= 2 for q in search_queries("hands typing on a laptop keyboard"))


def _video_result(link="https://cdn/x.mp4", user=None, page=None):
    video = {
        "video_files": [{"link": link, "file_type": "video/mp4", "width": 1080, "height": 1920}]
    }
    if user:
        video["user"] = {"name": user}
    if page:
        video["url"] = page
    return FakeResponse(json_data={"videos": [video]})


def test_an_empty_search_is_retried_with_a_shorter_query(tmp_path):
    session = FakeSession(
        FakeResponse(json_data={"videos": []}),
        _video_result(),
        FakeResponse(content=b"found-on-retry"),
    )
    provider = PexelsVisualProvider("key", session=session)

    asset = provider.fetch("person counting cash at kitchen table", tmp_path, scene_index=0)

    assert asset.source == "pexels"
    assert Path(asset.path).read_bytes() == b"found-on-retry"
    assert session.calls[0][1]["query"] == "person counting cash at kitchen table"
    assert session.calls[1][1]["query"] == "person counting cash"


def test_it_stops_after_the_allowed_number_of_queries_and_uses_a_card(tmp_path):
    empty = FakeResponse(json_data={"videos": []})
    session = FakeSession(empty, empty, empty)
    provider = PexelsVisualProvider("key", session=session)

    asset = provider.fetch("person counting cash at kitchen table", tmp_path, scene_index=0)

    assert asset.source == "generated"
    assert len(session.calls) == 3


def test_a_network_failure_is_not_retried_with_shorter_queries(tmp_path):
    import requests

    session = FakeSession(requests.ConnectionError("down"))
    provider = PexelsVisualProvider("key", session=session)

    asset = provider.fetch("person counting cash at kitchen table", tmp_path, scene_index=0)

    assert asset.source == "generated"
    assert len(session.calls) == 1


# --- crediting the footage -------------------------------------------------------


def test_pexels_footage_is_credited_to_its_creator_with_a_link(tmp_path):
    session = FakeSession(
        _video_result(user="Jane Doe", page="https://www.pexels.com/video/coins-123/"),
        FakeResponse(),
    )

    asset = PexelsVisualProvider("key", session=session).fetch("coins", tmp_path, scene_index=0)

    assert asset.credit == "Video by Jane Doe on Pexels: https://www.pexels.com/video/coins-123/"


def test_a_pexels_credit_still_links_to_pexels_when_the_response_has_no_details(tmp_path):
    """Pexels' API terms require a link back, so there is always one."""
    session = FakeSession(_video_result(), FakeResponse())

    asset = PexelsVisualProvider("key", session=session).fetch("coins", tmp_path, scene_index=0)

    assert asset.credit == "Video on Pexels: https://www.pexels.com"


def test_pixabay_footage_is_credited_too(tmp_path):
    search = FakeResponse(
        json_data={
            "hits": [
                {
                    "user": "someone",
                    "pageURL": "https://pixabay.com/videos/id-42/",
                    "videos": {
                        "large": {"url": "https://cdn/l.mp4", "width": 1080, "height": 1920}
                    },
                }
            ]
        }
    )
    session = FakeSession(search, FakeResponse())

    asset = PixabayVisualProvider("key", session=session).fetch("dogs", tmp_path, scene_index=0)

    assert asset.credit == "Video by someone on Pixabay: https://pixabay.com/videos/id-42/"


def test_a_generated_card_needs_no_credit(tmp_path):
    asset = GeneratedVisualProvider(width=32, height=32).fetch("cats", tmp_path, scene_index=0)

    assert asset.credit is None


def test_credit_line_without_an_author():
    assert credit_line("Pexels", None, "https://p/1", "https://home") == (
        "Video on Pexels: https://p/1"
    )


# --- which file to download --------------------------------------------------------


def renditions(*sizes):
    return [Rendition(f"https://cdn/{w}x{h}.mp4", w, h) for w, h in sizes]


def test_a_landscape_clip_gets_the_smallest_file_that_is_still_sharp_enough():
    """1080 lines cropped to the middle 9:16 is plenty; 4K is ten times the download."""
    chosen = best_rendition(
        renditions((3840, 2160), (1920, 1080), (1280, 720), (960, 540)),
        frame_width=1080,
        frame_height=1920,
    )

    assert (chosen.width, chosen.height) == (1920, 1080)


def test_a_portrait_clip_gets_the_smallest_file_that_fills_the_frame():
    chosen = best_rendition(
        renditions((2160, 3840), (1080, 1920), (720, 1280)), frame_width=1080, frame_height=1920
    )

    assert (chosen.width, chosen.height) == (1080, 1920)


def test_when_nothing_is_big_enough_the_largest_file_is_used():
    chosen = best_rendition(
        renditions((640, 360), (1280, 720), (960, 540)), frame_width=1080, frame_height=1920
    )

    assert (chosen.width, chosen.height) == (1280, 720)


def test_no_renditions_means_no_choice():
    assert best_rendition([], frame_width=1080, frame_height=1920) is None


def test_the_first_portrait_clip_beats_an_earlier_landscape_one():
    landscape = (Rendition("https://cdn/l.mp4", 1920, 1080), "landscape credit")
    portrait = (Rendition("https://cdn/p.mp4", 1080, 1920), "portrait credit")

    assert first_vertical_else_first([landscape, portrait]) == portrait


def test_without_a_portrait_clip_the_engines_top_result_is_used():
    first = (Rendition("https://cdn/1.mp4", 1920, 1080), "first")
    second = (Rendition("https://cdn/2.mp4", 1920, 1080), "second")

    assert first_vertical_else_first([first, second]) == first
    assert first_vertical_else_first([]) is None


def test_pixabay_downloads_the_1080p_file_not_the_4k_one(tmp_path):
    search = FakeResponse(
        json_data={
            "hits": [
                {
                    "videos": {
                        "large": {"url": "https://cdn/4k.mp4", "width": 3840, "height": 2160},
                        "medium": {"url": "https://cdn/1080.mp4", "width": 1920, "height": 1080},
                        "small": {"url": "https://cdn/720.mp4", "width": 1280, "height": 720},
                        "tiny": {"url": "https://cdn/540.mp4", "width": 960, "height": 540},
                    }
                }
            ]
        }
    )
    session = FakeSession(search, FakeResponse(content=b"1080p"))

    PixabayVisualProvider("key", session=session).fetch("dogs", tmp_path, 0)

    assert session.calls[1][0] == "https://cdn/1080.mp4"


def test_pixabay_skips_renditions_it_does_not_offer(tmp_path):
    """The API leaves "large" empty (url "", size 0) for clips with no 4K version."""
    search = FakeResponse(
        json_data={
            "hits": [
                {
                    "videos": {
                        "large": {"url": "", "width": 0, "height": 0},
                        "medium": {"url": "https://cdn/1080.mp4", "width": 1920, "height": 1080},
                    }
                }
            ]
        }
    )
    session = FakeSession(search, FakeResponse())

    asset = PixabayVisualProvider("key", session=session).fetch("dogs", tmp_path, 0)

    assert asset.source == "pixabay"
    assert session.calls[1][0] == "https://cdn/1080.mp4"


def test_pixabay_asks_for_real_footage_not_animations(tmp_path):
    session = FakeSession(FakeResponse(json_data={"hits": []}))

    PixabayVisualProvider("key", session=session).fetch("dogs", tmp_path, 0)

    assert session.calls[0][1]["video_type"] == "film"


def test_the_engines_order_decides_between_landscape_clips(tmp_path):
    """Result 1 only has a 4K file and result 2 a 1080p one: result 1 is still the
    better match, so resolution must not reorder them."""
    search = FakeResponse(
        json_data={
            "hits": [
                {
                    "videos": {
                        "large": {"url": "https://cdn/first.mp4", "width": 3840, "height": 2160}
                    }
                },
                {
                    "videos": {
                        "medium": {"url": "https://cdn/second.mp4", "width": 1920, "height": 1080}
                    }
                },
            ]
        }
    )
    session = FakeSession(search, FakeResponse())

    PixabayVisualProvider("key", session=session).fetch("dogs", tmp_path, 0)

    assert session.calls[1][0] == "https://cdn/first.mp4"


def test_a_portrait_result_further_down_beats_a_landscape_one_above_it(tmp_path):
    search = FakeResponse(
        json_data={
            "videos": [
                {
                    "video_files": [
                        {
                            "link": "https://cdn/landscape.mp4",
                            "file_type": "video/mp4",
                            "width": 1920,
                            "height": 1080,
                        }
                    ]
                },
                {
                    "video_files": [
                        {
                            "link": "https://cdn/portrait.mp4",
                            "file_type": "video/mp4",
                            "width": 1080,
                            "height": 1920,
                        }
                    ]
                },
            ]
        }
    )
    session = FakeSession(search, FakeResponse())

    PexelsVisualProvider("key", session=session).fetch("cats", tmp_path, 0)

    assert session.calls[1][0] == "https://cdn/portrait.mp4"


# --- a response that is not what the documentation promised -------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"videos": "oops"},
        {"videos": [None]},
        {
            "videos": [
                {
                    "video_files": [
                        {"link": "x", "file_type": "video/mp4", "width": "w", "height": "h"}
                    ]
                }
            ]
        },
        {"videos": [{"video_files": "none"}]},
        {"error": "Too many requests"},
        [],
    ],
)
def test_a_malformed_response_falls_back_instead_of_crashing_the_run(tmp_path, body, caplog):
    provider = PexelsVisualProvider("key", session=FakeSession(FakeResponse(json_data=body)))

    with caplog.at_level("INFO"):
        asset = provider.fetch("cats", tmp_path, 0)

    assert asset.source == "generated"
