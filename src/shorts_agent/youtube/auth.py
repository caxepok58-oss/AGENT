"""YouTube OAuth for the installed-app flow.

Uploading requires OAuth — an API key is not sufficient, because uploads act on
behalf of a channel. The token is cached on disk and refreshed automatically, so
the browser consent step happens once per machine.

Scope note: ``youtube.upload`` alone can insert videos but cannot read channel
data or add to playlists, so the full ``youtube`` scope is requested when those
are needed. Grant the narrowest scope your workflow actually uses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shorts_agent.exceptions import ConfigError

logger = logging.getLogger(__name__)

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
FULL_SCOPE = "https://www.googleapis.com/auth/youtube"
READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"

DEFAULT_SCOPES = [UPLOAD_SCOPE, FULL_SCOPE]


def _imports():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ConfigError(
            "Google API client libraries are required for YouTube access. "
            "Install with: pip install 'shorts-agent[youtube]'"
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def run_oauth_flow(
    client_secrets_file: str | Path,
    token_file: str | Path,
    scopes: list[str] | None = None,
    *,
    port: int = 0,
) -> Any:
    """Run the consent flow and cache credentials to ``token_file``."""
    _, _, InstalledAppFlow, _ = _imports()

    secrets_path = Path(client_secrets_file)
    if not secrets_path.exists():
        raise ConfigError(
            f"OAuth client secrets not found at {secrets_path}. Create an OAuth client "
            "of type 'Desktop app' in Google Cloud Console and download the JSON — "
            "see docs/SETUP.md."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes or DEFAULT_SCOPES)
    credentials = flow.run_local_server(port=port, prompt="consent")
    _save(credentials, Path(token_file))
    logger.info("Saved YouTube credentials to %s", token_file)
    return credentials


def load_credentials(token_file: str | Path, scopes: list[str] | None = None) -> Any | None:
    """Load cached credentials, refreshing them if expired."""
    Request, Credentials, _, _ = _imports()

    path = Path(token_file)
    if not path.exists():
        return None

    credentials = Credentials.from_authorized_user_file(str(path), scopes or DEFAULT_SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        logger.info("Refreshing expired YouTube credentials")
        credentials.refresh(Request())
        _save(credentials, path)
    return credentials


def get_youtube_service(
    client_secrets_file: str | Path,
    token_file: str | Path,
    scopes: list[str] | None = None,
    *,
    interactive: bool = False,
) -> Any:
    """Return an authenticated YouTube Data API client.

    ``interactive`` controls what happens when no valid token exists: the CLI's
    ``auth`` command opens a browser, while automated runs fail with a clear
    message instead of blocking on a consent screen nobody is watching.
    """
    _, _, _, build = _imports()

    credentials = load_credentials(token_file, scopes)
    if not credentials or not credentials.valid:
        if not interactive:
            raise ConfigError(
                f"No valid YouTube credentials at {token_file}. "
                "Run `shorts-agent auth` once to authorize this machine."
            )
        credentials = run_oauth_flow(client_secrets_file, token_file, scopes)

    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def _save(credentials: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json())
    # The token grants channel write access, so keep it owner-only.
    try:
        path.chmod(0o600)
    except OSError:  # noqa: PERF203 - not all filesystems support chmod
        logger.debug("Could not restrict permissions on %s", path)
