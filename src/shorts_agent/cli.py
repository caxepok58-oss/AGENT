"""Command-line interface.

Safety shape of this CLI: producing a video and publishing it are separate
commands, and ``run`` renders without uploading unless ``--publish`` is passed.
That is deliberate — see docs/POLICY.md.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from shorts_agent.__about__ import __version__
from shorts_agent.config import AppConfig, load_config
from shorts_agent.exceptions import ShortsAgentError
from shorts_agent.models import GeneratedVideo, Idea, Script, VideoMetadata
from shorts_agent.pipeline import Pipeline

app = typer.Typer(
    add_completion=False,
    help="Research trends, produce YouTube Shorts, and publish them.",
    no_args_is_help=True,
)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )
    # These libraries log every HTTP request at INFO, which drowns out our own
    # progress messages.
    for noisy in ("httpx", "urllib3", "googleapiclient.discovery_cache", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config_path: str | None) -> AppConfig:
    try:
        return load_config(config_path)
    except ShortsAgentError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _fail(exc: Exception) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


ConfigOption = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
VerboseOption = typer.Option(False, "--verbose", "-v", help="Enable debug logging.")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"shorts-agent {__version__}")


@app.command()
def init(
    directory: Path | None = typer.Option(
        None, "--dir", "-d", help="Where to write config files (default: current directory)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Copy the example config and .env into place."""
    # Resolved here rather than as a default so it reflects the invocation's
    # working directory, not the process's directory at import time.
    directory = directory or Path.cwd()
    pairs = [
        (PROJECT_ROOT / "config" / "config.example.yaml", directory / "config" / "config.yaml"),
        (PROJECT_ROOT / ".env.example", directory / ".env"),
        (
            PROJECT_ROOT / "config" / "manual_trends.example.yaml",
            directory / "config" / "manual_trends.yaml",
        ),
        (
            PROJECT_ROOT / "config" / "policy_blocklist.example.yaml",
            directory / "config" / "policy_blocklist.yaml",
        ),
    ]

    for source, target in pairs:
        if not source.exists():
            console.print(f"[yellow]Skipped[/yellow] {target.name}: {source} not found")
            continue
        if target.exists() and not force:
            console.print(f"[yellow]Exists[/yellow]  {target} (use --force to overwrite)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        console.print(f"[green]Created[/green] {target}")

    console.print(
        "\nNext: add your API keys to [bold].env[/bold], set the channel niche and "
        "persona in [bold]config/config.yaml[/bold], then run "
        "[bold]shorts-agent trends[/bold] to check your setup.\n"
        "Read [bold]docs/POLICY.md[/bold] before enabling publishing."
    )


@app.command()
def doctor(
    config: str | None = ConfigOption,
    verbose: bool = VerboseOption,
) -> None:
    """Check your setup and report what works, what will degrade, and what will fail."""
    _setup_logging(verbose)
    from shorts_agent.diagnostics import run_all, summarize

    checks = run_all(_load(config))

    symbols = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]"}
    for check in checks:
        console.print(f"{symbols[check.status]} {check.name}: {check.detail}")
        if check.fix and check.status != "ok":
            console.print(f"   [dim]→ {check.fix}[/dim]")

    ok, warn, fail = summarize(checks)
    console.print(f"\n{ok} ok, {warn} warning(s), {fail} problem(s)")

    if fail:
        console.print("[red]Some parts of the pipeline cannot run.[/red] Fix the ✗ items above.")
        raise typer.Exit(code=1)
    if warn:
        console.print(
            "[yellow]Everything runs, but some features will fall back.[/yellow] "
            "The ! items explain what you would gain by configuring them."
        )
    else:
        console.print("[green]Everything is configured.[/green]")


@app.command()
def trends(
    config: str | None = ConfigOption,
    limit: int = typer.Option(15, "--limit", "-n", help="Maximum topics to show."),
    verbose: bool = VerboseOption,
) -> None:
    """Show the current trend signals for your niche."""
    _setup_logging(verbose)
    pipeline = Pipeline(_load(config))

    try:
        topics = pipeline.research(limit=limit)
    except ShortsAgentError as exc:
        _fail(exc)
        return

    if not topics:
        console.print(
            "[yellow]No trend signals found.[/yellow] Set YOUTUBE_API_KEY for live "
            "signals, or add keywords to config/manual_trends.yaml."
        )
        return

    table = Table(title=f"Trend signals — {pipeline.config.channel.niche}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Topic")
    table.add_column("Score", justify="right")
    table.add_column("Sources")

    for i, topic in enumerate(topics, start=1):
        table.add_row(str(i), topic.keyword[:70], f"{topic.score:.1f}", topic.source)
    console.print(table)


@app.command()
def ideate(
    config: str | None = ConfigOption,
    count: int | None = typer.Option(None, "--count", "-n", help="How many ideas to generate."),
    verbose: bool = VerboseOption,
) -> None:
    """Generate video ideas from current trends, without producing anything."""
    _setup_logging(verbose)
    pipeline = Pipeline(_load(config))

    try:
        topics = pipeline.research()
        ideas = pipeline.ideate(topics, count=count)
    except ShortsAgentError as exc:
        _fail(exc)
        return

    guard = pipeline._guard()  # noqa: SLF001 - CLI surfaces the same check the run uses
    for i, idea in enumerate(ideas, start=1):
        check = guard.check_idea(idea)
        status = "[green]ok[/green]" if check.passed else "[red]blocked[/red]"
        console.print(f"\n[bold]{i}. {idea.title}[/bold]  {status}")
        console.print(f"   [dim]Hook:[/dim] {idea.hook}")
        console.print(f"   [dim]Premise:[/dim] {idea.premise}")
        console.print(f"   [dim]Why it should retain:[/dim] {idea.virality_reasoning}")
        if not check.passed:
            console.print(f"   [red]Reasons:[/red] {'; '.join(check.reasons)}")


@app.command()
def run(
    config: str | None = ConfigOption,
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Upload the finished video. Without this, the video is only rendered locally.",
    ),
    no_schedule: bool = typer.Option(
        False, "--no-schedule", help="Publish immediately instead of scheduling a slot."
    ),
    ideas: int | None = typer.Option(None, "--ideas", "-n", help="How many ideas to consider."),
    verbose: bool = VerboseOption,
) -> None:
    """Run the full pipeline: research, script, render, and optionally upload."""
    _setup_logging(verbose)
    pipeline = Pipeline(_load(config))

    try:
        result = pipeline.run(publish=publish, schedule=not no_schedule, idea_count=ideas)
    except ShortsAgentError as exc:
        _fail(exc)
        return

    console.print(f"\n[bold green]Run {result.run_id} complete[/bold green]")
    if result.idea:
        console.print(f"  Idea:     {result.idea.title}")
    if result.estimated_duration_seconds:
        console.print(f"  Duration: ~{result.estimated_duration_seconds:.0f}s")
    if result.metadata:
        console.print(f"  Title:    {result.metadata.title}")
        console.print(f"  Tags:     {', '.join(result.metadata.tags[:8])}")
    if result.video_path:
        console.print(f"  Video:    {result.video_path}")
    if result.publish:
        console.print(f"  URL:      {result.publish.url} ({result.publish.privacy_status})")
        if result.publish.publish_at:
            console.print(f"  Goes live: {result.publish.publish_at.isoformat()}")
    elif result.video_path:
        console.print(
            f"\n[yellow]Not uploaded.[/yellow] Review it with "
            f"[bold]shorts-agent preview {result.run_id}[/bold], then "
            f"[bold]shorts-agent publish {result.run_id}[/bold] to upload."
        )


@app.command()
def preview(
    run_id: str = typer.Argument(..., help="Run id from a previous `run` command."),
    config: str | None = ConfigOption,
    frames: int = typer.Option(6, "--frames", "-f", help="How many frames to sample."),
    thumbnail: bool = typer.Option(
        False, "--thumbnail", help="Also write a thumbnail candidate JPEG."
    ),
    verbose: bool = VerboseOption,
) -> None:
    """Review a rendered video: a contact sheet of frames, plus its script and metadata."""
    _setup_logging(verbose)
    from shorts_agent.video.frames import (
        FrameExtractionError,
        contact_sheet,
        probe_duration,
    )
    from shorts_agent.video.frames import thumbnail as make_thumbnail

    pipeline = Pipeline(_load(config))
    record = pipeline.storage.get_run(run_id)
    if not record:
        console.print(f"[red]No run found with id {run_id}[/red]")
        raise typer.Exit(code=1)

    video_path = Path(record["video_path"] or "")
    if not video_path.exists():
        console.print(
            f"[red]Run {run_id} has no rendered video[/red] (status: {record.get('status')})"
        )
        raise typer.Exit(code=1)

    if record.get("script_json"):
        script = Script.model_validate_json(record["script_json"])
        console.print(f"\n[bold]{script.title}[/bold]")
        for scene in script.scenes:
            console.print(f"  [dim]{scene.index + 1}.[/dim] {scene.text}")
            if scene.on_screen_text:
                console.print(f"     [dim]on screen:[/dim] {scene.on_screen_text}")

    if record.get("metadata_json"):
        metadata = VideoMetadata.model_validate_json(record["metadata_json"])
        console.print(f"\n[bold]Title:[/bold] {metadata.title}")
        console.print(f"[bold]Description:[/bold]\n{metadata.description}")
        console.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}")
        console.print(
            f"[bold]Disclosed as synthetic:[/bold] {metadata.contains_synthetic_media} | "
            f"[bold]Privacy:[/bold] {metadata.privacy_status}"
        )

    try:
        duration = probe_duration(video_path)
        console.print(f"\n[bold]Video:[/bold] {video_path} ({duration:.1f}s)")
        sheet = contact_sheet(video_path, video_path.parent / "contact_sheet.png", count=frames)
        console.print(f"[green]Contact sheet:[/green] {sheet}")
        if thumbnail:
            thumb = make_thumbnail(video_path, video_path.parent / "thumbnail.jpg")
            console.print(f"[green]Thumbnail:[/green] {thumb}")
    except FrameExtractionError as exc:
        console.print(f"[yellow]Could not build the contact sheet:[/yellow] {exc}")

    console.print(
        "\nWatch the video itself before publishing — a contact sheet cannot show "
        "audio, pacing, or caption timing."
    )


@app.command()
def publish(
    run_id: str = typer.Argument(..., help="Run id from a previous `run` command."),
    config: str | None = ConfigOption,
    no_schedule: bool = typer.Option(
        False, "--no-schedule", help="Publish immediately instead of scheduling a slot."
    ),
    verbose: bool = VerboseOption,
) -> None:
    """Upload a previously rendered video after you have reviewed it."""
    _setup_logging(verbose)
    pipeline = Pipeline(_load(config))

    record = pipeline.storage.get_run(run_id)
    if not record:
        console.print(f"[red]No run found with id {run_id}[/red]")
        raise typer.Exit(code=1)
    if not record.get("video_path") or not record.get("metadata_json"):
        console.print(
            f"[red]Run {run_id} has no rendered video or metadata[/red] "
            f"(status: {record.get('status')})"
        )
        raise typer.Exit(code=1)

    video = GeneratedVideo(
        run_id=run_id,
        video_path=record["video_path"],
        metadata=VideoMetadata.model_validate_json(record["metadata_json"]),
        script=Script.model_validate_json(record["script_json"]),
        idea=Idea(title=record["idea_title"] or "", hook="", premise=""),
    )

    try:
        result = pipeline.publish(video, schedule=not no_schedule)
    except ShortsAgentError as exc:
        _fail(exc)
        return

    console.print(f"[green]Uploaded:[/green] {result.url} ({result.privacy_status})")


@app.command()
def auth(
    config: str | None = ConfigOption,
    verbose: bool = VerboseOption,
) -> None:
    """Authorize this machine to upload to your YouTube channel."""
    _setup_logging(verbose)
    from shorts_agent.config import get_settings
    from shorts_agent.youtube import get_youtube_service

    _load(config)
    settings = get_settings()

    try:
        service = get_youtube_service(
            settings.youtube_client_secrets_file, settings.youtube_token_file, interactive=True
        )
        response = service.channels().list(part="snippet", mine=True).execute()
    except ShortsAgentError as exc:
        _fail(exc)
        return
    except Exception as exc:  # noqa: BLE001 - surface any Google client failure
        _fail(exc)
        return

    items = response.get("items", [])
    if items:
        console.print(f"[green]Authorized[/green] as channel: {items[0]['snippet']['title']}")
    else:
        console.print(
            "[yellow]Authorized, but no channel was returned.[/yellow] "
            "Make sure this Google account has a YouTube channel."
        )


@app.command()
def report(
    config: str | None = ConfigOption,
    refresh: bool = typer.Option(
        False, "--refresh", help="Fetch fresh view counts from YouTube first."
    ),
    limit: int = typer.Option(15, "--limit", "-n", help="How many runs to show."),
    verbose: bool = VerboseOption,
) -> None:
    """Show recent runs and how published videos are performing."""
    _setup_logging(verbose)
    pipeline = Pipeline(_load(config))

    if refresh:
        try:
            updated = pipeline.refresh_stats()
            console.print(f"[dim]Refreshed stats for {updated} videos[/dim]")
        except ShortsAgentError as exc:
            console.print(f"[yellow]Could not refresh stats:[/yellow] {exc}")

    runs = pipeline.storage.recent_runs(limit=limit)
    if not runs:
        console.print("No runs recorded yet.")
        return

    table = Table(title="Recent runs")
    table.add_column("Run", style="dim")
    table.add_column("When", style="dim")
    table.add_column("Status")
    table.add_column("Idea")

    for record in runs:
        status = record["status"] or ""
        colour = {"published": "green", "failed": "red"}.get(status, "white")
        table.add_row(
            record["run_id"],
            (record["created_at"] or "")[:16],
            f"[{colour}]{status}[/{colour}]",
            (record["idea_title"] or "")[:50],
        )
    console.print(table)

    performers = pipeline.storage.top_performers(limit=5)
    if performers:
        best = Table(title="Best performing")
        best.add_column("Title")
        best.add_column("Views", justify="right")
        best.add_column("Likes", justify="right")
        for row in performers:
            best.add_row((row["title"] or "")[:50], str(row["views"] or 0), str(row["likes"] or 0))
        console.print(best)


@app.command()
def daemon(
    config: str | None = ConfigOption,
    interval_hours: float = typer.Option(12.0, "--every", help="Hours between runs."),
    publish: bool = typer.Option(False, "--publish", help="Upload each finished video."),
    verbose: bool = VerboseOption,
) -> None:
    """Run the pipeline on a schedule.

    The configured daily upload cap still applies, so a short interval will not
    publish more than that cap allows.
    """
    _setup_logging(verbose)
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        console.print(
            "[red]APScheduler is required for the daemon.[/red] "
            "Install with: pip install 'shorts-agent[scheduler]'"
        )
        raise typer.Exit(code=2) from None

    pipeline = Pipeline(_load(config))

    def job() -> None:
        try:
            result = pipeline.run(publish=publish)
            console.print(f"[green]Run {result.run_id} finished[/green]")
        except ShortsAgentError as exc:
            # Keep the daemon alive: a single failed run (rate limit, provider
            # outage, policy block) should not stop the schedule.
            console.print(f"[red]Run failed:[/red] {exc}")

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", hours=interval_hours, next_run_time=None)
    console.print(
        f"Daemon started: a run every {interval_hours}h "
        f"(publish={'on' if publish else 'off'}). Ctrl-C to stop."
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print("\nDaemon stopped.")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
