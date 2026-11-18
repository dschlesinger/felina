import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from data_tools import leaderboard_lit
from data_tools.leaderboard_lit import (
    GitTarget,
    TooLarge,
    clone,
    fetch_target,
    google_export,
    latest_snapshots,
    parse_git,
    parse_html,
    read_snapshots,
    write_reports,
)


@pytest.mark.parametrize(
    "url, clone_url, ref, path",
    [
        ("https://github.com/o/r", "https://github.com/o/r.git", None, ""),
        ("https://github.com/o/r.git", "https://github.com/o/r.git", None, ""),
        (
            "https://github.com/o/r/blob/main/docs/report.md",
            "https://github.com/o/r.git",
            "main",
            "docs/report.md",
        ),
        (
            "https://github.com/o/r/tree/" + "a" * 40,
            "https://github.com/o/r.git",
            "a" * 40,
            "",
        ),
        (
            "https://gist.github.com/u/0123abcd",
            "https://gist.github.com/0123abcd.git",
            None,
            "",
        ),
        ("https://huggingface.co/o/m", "https://huggingface.co/o/m", None, ""),
        (
            "https://huggingface.co/o/m/blob/main/README.md",
            "https://huggingface.co/o/m",
            "main",
            "README.md",
        ),
        (
            "https://huggingface.co/spaces/o/s/tree/main",
            "https://huggingface.co/spaces/o/s",
            "main",
            "",
        ),
        (
            "https://tangled.org/h.bsky.social/r/blob/abc1234/x/REPORT.md",
            "https://tangled.org/h.bsky.social/r",
            "abc1234",
            "x/REPORT.md",
        ),
        (
            "https://gitlab.com/g/sub/r/-/blob/dev/a.md",
            "https://gitlab.com/g/sub/r.git",
            "dev",
            "a.md",
        ),
        (
            "https://codeberg.org/o/r/src/branch/main/a.md",
            "https://codeberg.org/o/r.git",
            "main",
            "a.md",
        ),
        (
            "https://example.org/some/repo.git",
            "https://example.org/some/repo.git",
            None,
            "",
        ),
    ],
)
def test_parse_git_recognises_forges(url, clone_url, ref, path):
    target = parse_git(url)
    assert (target.clone_url, target.ref, target.path) == (clone_url, ref, path)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/orgs/foo",
        "https://huggingface.co/papers/1234.5678",
        "https://docs.google.com/document/d/abc/edit",
        "https://gist.githubusercontent.com/u/id/raw/sha/file.md",
        "https://example.org/blog/post",
    ],
)
def test_parse_git_leaves_other_links(url):
    assert parse_git(url) is None


def test_pinned_and_unpinned_links_into_one_repo_get_different_folders():
    sha = "b" * 40
    assert (
        parse_git(f"https://github.com/o/r/tree/{sha}").key
        != parse_git("https://github.com/o/r").key
    )
    assert (
        parse_git("https://github.com/o/r/blob/main/a.md").key
        == parse_git("https://github.com/o/r").key
    )


def test_google_export():
    doc = "1" * 30
    assert google_export(f"https://docs.google.com/document/d/{doc}/edit?usp=sharing")[
        0
    ] == (f"https://docs.google.com/document/d/{doc}/export?format=txt")
    assert google_export(f"https://drive.google.com/file/d/{doc}/view") == [
        f"https://drive.google.com/uc?export=download&id={doc}"
    ]
    assert (
        google_export(
            "https://docs.google.com/document/d/e/2PACX-1vQabcdefghijklmnopqrstuvwxyz/pub"
        )
        == []
    )
    assert google_export("https://example.org/d/" + doc) == []


def test_parse_html_extracts_title_text_and_links():
    html = """<html><head><title>Report</title><style>p{color:red}</style></head>
    <body><h2>Method</h2><p>We used   <a href="/code">Chemprop</a>.</p>
    <script>var x = 1;</script><ul><li><a href="https://x.org/a"><span>first</span></a></li></ul></body></html>"""
    title, text, links = parse_html(html, "https://site.org/post/")
    assert title == "Report"
    assert "## Method" in text and "We used Chemprop." in text and "- first" in text
    assert "color" not in text and "var x" not in text
    assert links == ["https://site.org/code", "https://x.org/a"]


def test_read_snapshots_collects_links_per_user(tmp_path):
    header = "Rank,Username,Model Report Link,Open Code\n"
    (tmp_path / "board_a__20260101T000000Z.csv").write_text(
        header + "1,alice,https://github.com/a/r,Yes\n2,bob,Not submitted,No\n"
        "3,carol,see https://x.org/a and https://x.org/b.,No\n"
    )
    (tmp_path / "board_b__20260101T000000Z.csv").write_text(
        header + "1,alice,https://github.com/a/r,Yes\n"
    )
    links = read_snapshots(sorted(tmp_path.glob("*.csv")))
    assert set(links) == {
        ("alice", "https://github.com/a/r"),
        ("carol", "https://x.org/a"),
        ("carol", "https://x.org/b"),
    }
    assert links["alice", "https://github.com/a/r"].boards == {"board_a", "board_b"}


def test_latest_snapshots_keeps_newest_per_board(tmp_path):
    for name in ("a__20260101T000000Z", "a__20260201T000000Z", "b__20260101T000000Z"):
        (tmp_path / f"{name}.csv").write_text("")
    assert [p.name for p in latest_snapshots(tmp_path)] == [
        "a__20260201T000000Z.csv",
        "b__20260101T000000Z.csv",
    ]


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_clone_pins_annotated_tag_and_skips_large_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "small.md").write_text("report v1\n")
    (src / "big.bin").write_bytes(b"x" * 2000)
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "v1")
    _git(src, "tag", "-a", "v1", "-m", "release")
    (src / "small.md").write_text("report v2\n")
    _git(src, "commit", "-qam", "v2")
    _git(src, "config", "uploadpack.allowFilter", "true")

    dest = tmp_path / "out"
    clone(GitTarget(f"file://{src}", "v1", "small.md", "k"), dest, max_bytes=1000)
    assert (dest / "repo" / "small.md").read_text() == "report v1\n"
    assert not (dest / "repo" / "big.bin").exists()
    assert "big.bin" in (dest / "LARGE_FILES_SKIPPED.txt").read_text()


def test_write_reports_lists_failed_and_thin(tmp_path):
    row = dict.fromkeys(
        [
            "user",
            "link",
            "status",
            "handler",
            "folder",
            "fetched",
            "path",
            "detail",
            "boards",
            "open_code",
        ],
        "",
    )
    rows = [
        {**row, "user": "a", "status": "ok"},
        {**row, "user": "b", "status": "failed", "detail": "HTTP 403 x|y"},
        {**row, "user": "c", "status": "thin"},
    ]
    write_reports(tmp_path, rows)
    report = (tmp_path / "FAILURES.md").read_text()
    assert (
        "| b | failed |" in report
        and "HTTP 403 x/y" in report
        and "| c | thin |" in report
    )
    assert "| a |" not in report
    assert (tmp_path / "index.csv").read_text().count("\n") == 4


def test_clone_aborts_over_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(leaderboard_lit, "MAX_DOWNLOAD_BYTES", 100_000)
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    for i in range(
        3
    ):  # small files each, too big together (random bytes do not compress)
        (src / f"part{i}.bin").write_bytes(os.urandom(60_000))
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "v1")

    dest = tmp_path / "out"
    with pytest.raises(TooLarge):
        clone(GitTarget(f"file://{src}", None, "", "k"), dest, max_bytes=100_000)
    result = fetch_target(
        "git",
        GitTarget(f"file://{src}", None, "", "k"),
        [],
        tmp_path / "out2",
        100_000,
        False,
    )
    assert result.status == "failed" and "limit" in result.detail
    assert not (tmp_path / "out2" / "repo").exists()


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"%PDF" + b"x" * 5000
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            if self.path == "/sized":
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.mark.parametrize("path", ["/sized", "/streamed"])
def test_downloads_over_the_size_cap_are_refused(server, tmp_path, monkeypatch, path):
    monkeypatch.setattr(leaderboard_lit, "MAX_DOWNLOAD_BYTES", 1000)
    result = fetch_target("http", None, [server + path], tmp_path, None, False)
    assert result.status == "failed" and "limit" in result.detail
    assert not any(tmp_path.iterdir())


def test_downloads_under_the_size_cap_are_saved(server, tmp_path):
    result = fetch_target("http", None, [server + "/streamed"], tmp_path, None, False)
    assert result.status == "ok"
    assert (tmp_path / "page.pdf").stat().st_size == 5004
