"""Download the method reports and code linked from the challenge leaderboards, for a literature review.

Reads the leaderboard snapshots saved by `download-leaderboard` (the latest one of each leaderboard, or the CSVs given
on the command line, or a fresh download with `--fetch`), collects every URL in the `Model Report Link` column, and
fetches each one into `<RESULTS_FOLDER>/leaderboard_lit/<space>/<user>/<target>/` (`--out` overrides the root):

  * git repositories (GitHub, gists, Hugging Face models/datasets/spaces, tangled, GitLab, Codeberg, Bitbucket, any
    `*.git` URL, and any other URL that answers `git ls-remote`) are shallow-cloned into `repo/`, pinned to the commit
    or branch in the link; the file or folder the link points at is recorded as `path`. Files over `--max-mb` are left
    out of the checkout and listed in `LARGE_FILES_SKIPPED.txt`.
  * Google Docs are exported as plain text, Google Drive files are downloaded directly.
  * anything else is downloaded; HTML is saved as `page.html` and parsed into `page.txt` (readable text) and
    `links.txt` (every link on the page), other content types are saved as `page.<ext>`.

Each target folder gets a `source.txt` listing the links and leaderboards it came from. `index.csv` has one row per
link with its status, and `FAILURES.md` lists every link that could not be fetched (HTTP 403/404, timeouts, failed
clones) or came back as a near-empty page (`thin`: usually a login wall or a JavaScript-only site). Reruns reuse what
is already on disk and retry only what failed (`--refetch` fetches everything again).
"""

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from data_tools.download_leaderboard import (
    URL_ENV,
    download,
    find_leaderboards,
    resolve_base_url,
    snapshot_dir,
)
from data_tools.paths import RESULTS_DIR

LINK_COLUMN = "Model Report Link"
SHA = re.compile(r"^[0-9a-f]{7,40}$")
URL_RE = re.compile(r"https?://[^\s,;<>\"']+")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) leaderboard-lit"
MAX_PAGE_BYTES = 50_000_000
THIN_CHARS = 200  # parsed pages with less text than this are flagged as `thin`
GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_LFS_SKIP_SMUDGE": "1",
    "GIT_ASKPASS": "true",
}
EXTENSIONS = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/zip": ".zip",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
# First path segments of github.com / huggingface.co that are site pages rather than repositories.
GITHUB_RESERVED = {
    "orgs",
    "settings",
    "features",
    "marketplace",
    "topics",
    "collections",
    "sponsors",
    "users",
    "about",
}
HF_RESERVED = {
    "papers",
    "docs",
    "blog",
    "collections",
    "organizations",
    "settings",
    "learn",
    "posts",
    "models",
    "tasks",
}


@dataclass(frozen=True)
class GitTarget:
    clone_url: str
    ref: str | None  # branch, tag or commit named in the link
    path: str  # file or folder inside the repository the link points at
    key: str  # folder name for the clone


@dataclass
class Link:
    url: str
    user: str
    boards: set[str] = field(default_factory=set)
    open_code: set[str] = field(default_factory=set)


@dataclass
class Result:
    status: str  # ok | cached | thin | failed
    handler: str
    fetched: str = ""
    path: str = ""
    detail: str = ""


# ---------------------------------------------------------------- link collection


def read_snapshots(paths: list[Path]) -> dict[tuple[str, str], Link]:
    """Every (user, url) pair in the `Model Report Link` column of the given leaderboard CSVs."""
    links: dict[tuple[str, str], Link] = {}
    for p in paths:
        board = p.stem.split("__")[0]
        for row in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"))):
            user = (row.get("Username") or "").strip()
            for url in URL_RE.findall(row.get(LINK_COLUMN) or ""):
                url = url.rstrip(").,;]")
                link = links.setdefault((user, url), Link(url, user))
                link.boards.add(board)
                link.open_code.add((row.get("Open Code") or "").strip())
    return links


def latest_snapshots(folder: Path) -> list[Path]:
    """The newest snapshot of each leaderboard in a `download-leaderboard` folder."""
    newest: dict[str, Path] = {}
    for p in sorted(folder.glob("*.csv")):
        newest[p.stem.split("__")[0]] = p
    return sorted(newest.values())


# ---------------------------------------------------------------- recognising git repositories


def _clean(part: str) -> str:
    return re.sub(r"\.git$", "", part)


def _key(*parts: str | None) -> str:
    return slug("_".join(p for p in parts if p))


def parse_git(url: str) -> GitTarget | None:
    """Recognise links into a known git host; None for anything else."""
    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.")
    parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
    if u.path.endswith(".git") and parts:
        return GitTarget(
            url.split("#")[0].split("?")[0], None, "", _key(host, *parts[-2:])
        )

    def blob_tree(
        owner: str, repo: str, rest: list[str], clone: str, markers=("blob", "tree")
    ) -> GitTarget:
        ref = path = None
        if len(rest) >= 2 and rest[0] in markers:
            ref, path = rest[1], "/".join(rest[2:])
        return GitTarget(
            clone,
            ref,
            path or "",
            _key(host, owner, _clean(repo), ref if ref and SHA.match(ref) else None),
        )

    if host == "github.com" and len(parts) >= 2 and parts[0] not in GITHUB_RESERVED:
        owner, repo = parts[0], _clean(parts[1])
        return blob_tree(
            owner, repo, parts[2:], f"https://github.com/{owner}/{repo}.git"
        )
    if host == "gist.github.com" and parts:
        gist = _clean(parts[-1] if len(parts) <= 2 else parts[1])
        return GitTarget(
            f"https://gist.github.com/{gist}.git", None, "", _key("gist", gist)
        )
    if host == "huggingface.co" and parts:
        kind = parts[0] if parts[0] in ("spaces", "datasets") else ""
        repo_parts = parts[1:] if kind else parts
        if len(repo_parts) < 2 or (not kind and parts[0] in HF_RESERVED):
            return None
        owner, repo = repo_parts[0], _clean(repo_parts[1])
        prefix = f"{kind}/" if kind else ""
        return blob_tree(
            owner,
            repo,
            repo_parts[2:],
            f"https://huggingface.co/{prefix}{owner}/{repo}",
            markers=("blob", "tree", "resolve"),
        )
    if host in ("tangled.org", "tangled.sh") and len(parts) >= 2:
        owner, repo = parts[0], _clean(parts[1])
        return blob_tree(owner, repo, parts[2:], f"https://{host}/{owner}/{repo}")
    if host == "gitlab.com" and len(parts) >= 2:
        project = parts[: parts.index("-")] if "-" in parts else parts
        rest = parts[len(project) + 1 :] if "-" in parts else []
        project[-1] = _clean(project[-1])
        return blob_tree(
            project[0],
            "_".join(project[1:]),
            rest,
            f"https://gitlab.com/{'/'.join(project)}.git",
        )
    if host == "codeberg.org" and len(parts) >= 2:
        owner, repo, rest = parts[0], _clean(parts[1]), parts[2:]
        if len(rest) >= 3 and rest[0] == "src":  # src/branch|commit|tag/<ref>/<path>
            rest = ["tree", *rest[2:]]
        return blob_tree(owner, repo, rest, f"https://codeberg.org/{owner}/{repo}.git")
    if host == "bitbucket.org" and len(parts) >= 2:
        owner, repo, rest = parts[0], _clean(parts[1]), parts[2:]
        return blob_tree(
            owner,
            repo,
            ["tree", *rest[1:]] if rest[:1] == ["src"] else rest,
            f"https://bitbucket.org/{owner}/{repo}.git",
        )
    return None


def probe_git(url: str) -> GitTarget | None:
    """Treat an unrecognised URL as a git repository if it answers `git ls-remote`."""
    clean = url.split("#")[0].split("?")[0].rstrip("/")
    try:
        subprocess.run(
            ["git", "ls-remote", "--exit-code", clean, "HEAD"],
            env=GIT_ENV,
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    u = urlparse(clean)
    return GitTarget(
        clean, None, "", _key(u.netloc, *u.path.strip("/").split("/")[-2:])
    )


# ---------------------------------------------------------------- cloning


def _git(
    repo_dir: Path, *args: str, timeout: int = 600, **kw
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        env=GIT_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
        **kw,
    )


def _checkout_small(repo_dir: Path, max_bytes: int) -> int:
    """Check out only files of at most max_bytes; list the others in LARGE_FILES_SKIPPED.txt. Returns files skipped."""
    tree = _git(repo_dir, "ls-tree", "-r", "-l", "-z", "HEAD").stdout
    keep, skipped = [], []
    for entry in filter(None, tree.split("\0")):
        meta, path = entry.split("\t", 1)
        size = meta.split()[3]
        if size != "-" and int(size) > max_bytes:
            skipped.append(f"{int(size) / 1e6:.1f}M\t{path}")
        else:
            keep.append(path)
    if keep:
        _git(
            repo_dir,
            "checkout",
            "-q",
            "HEAD",
            "--pathspec-from-file=-",
            "--pathspec-file-nul",
            input="\0".join(keep),
        )
    if skipped:
        (repo_dir.parent / "LARGE_FILES_SKIPPED.txt").write_text(
            "\n".join(skipped) + "\n"
        )
    return len(skipped)


def clone(target: GitTarget, dest: Path, max_bytes: int | None) -> str:
    """Shallow-clone target into dest/repo at the linked ref; returns a short description of what was checked out."""
    repo_dir = dest / "repo"
    repo_dir.mkdir(parents=True)
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "remote", "add", "origin", target.clone_url)
    filt = ["--filter", f"blob:limit={max_bytes + 1}"] if max_bytes else []
    ref = target.ref
    try:
        _git(repo_dir, "fetch", "-q", "--depth", "1", *filt, "origin", ref or "HEAD")
    except subprocess.CalledProcessError:
        if ref is None:
            raise
        # A branch name containing "/" is cut short by the blob/tree link format: fall back to the default branch.
        ref = None
        _git(repo_dir, "fetch", "-q", "--depth", "1", *filt, "origin", "HEAD")
    _git(
        repo_dir, "update-ref", "HEAD", "FETCH_HEAD^{commit}"
    )  # ^{commit} peels annotated tags
    commit = _git(repo_dir, "rev-parse", "--short", "HEAD").stdout.strip()
    skipped = _checkout_small(repo_dir, max_bytes) if max_bytes else 0
    if not max_bytes:
        _git(repo_dir, "checkout", "-q", "HEAD", "--", ".")
    note = f" ({skipped} large files skipped)" if skipped else ""
    used = ref or "default branch"
    return f"{target.clone_url} @ {used} {commit}{note}"


# ---------------------------------------------------------------- web pages


class _PageText(HTMLParser):
    """Readable text, title and links of an HTML page, using only the standard library."""

    SKIP = {"script", "style", "noscript", "template", "svg", "iframe"}
    BLOCK = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "section",
        "article",
        "pre",
        "blockquote",
        "table",
        "ul",
        "ol",
        "header",
        "footer",
        "hr",
        "main",
        "nav",
        "figure",
        "figcaption",
        "dt",
        "dd",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url, self.parts, self.links, self.title = base_url, [], [], ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif (
            tag == "a"
            and attrs.get("href")
            and not attrs["href"].startswith(("#", "javascript:", "mailto:"))
        ):
            self.links.append(urljoin(self.base_url, attrs["href"]))
        if tag in self.BLOCK:
            self.parts.append("\n")
        if re.fullmatch(r"h[1-6]", tag):
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (
            re.sub(r"[ \t\r\f\v]+", " ", line).strip()
            for line in "".join(self.parts).split("\n")
        )
        text = re.sub(
            r"^(#+|-)\n+(?=\S)", r"\1 ", "\n".join(lines), flags=re.MULTILINE
        )  # marker left alone by a nested block
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """(title, readable text, unique links) of an HTML page."""
    parser = _PageText(base_url)
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text(), list(dict.fromkeys(parser.links))


def http_get(url: str) -> tuple[bytes, str, str, str, bool]:
    """(body, media type, charset, attachment filename, truncated) of a GET request."""
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read(MAX_PAGE_BYTES + 1)
        media = (r.headers.get_content_type() or "application/octet-stream").lower()
        charset = r.headers.get_content_charset() or ""
        filename = r.headers.get_filename() or ""
    return body[:MAX_PAGE_BYTES], media, charset, filename, len(body) > MAX_PAGE_BYTES


def save_page(url: str, dest: Path) -> tuple[str, str, bool]:
    """Download url into dest; returns (files written, detail, thin)."""
    body, media, charset, filename, truncated = http_get(url)
    dest.mkdir(parents=True, exist_ok=True)
    if media in ("text/html", "application/xhtml+xml") or body.lstrip()[
        :15
    ].lower().startswith((b"<!doctype html", b"<html")):
        if not charset:
            m = re.search(
                rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.IGNORECASE
            )
            charset = m.group(1).decode() if m else "utf-8"
        html = body.decode(charset, errors="replace")
        title, text, links = parse_html(html, url)
        (dest / "page.html").write_bytes(body)
        (dest / "page.txt").write_text(
            (f"# {title}\n\n" if title else "") + text + "\n"
        )
        (dest / "links.txt").write_text("".join(f"{link}\n" for link in links))
        detail = (
            f"{title[:80]!r}, {len(text)} chars, {len(links)} links"
            if title
            else f"{len(text)} chars"
        )
        return "page.html, page.txt, links.txt", detail, len(text) < THIN_CHARS
    ext = (
        EXTENSIONS.get(media)
        or Path(filename).suffix
        or (".pdf" if body.startswith(b"%PDF") else "")
        or Path(urlparse(url).path).suffix
        or ".bin"
    )[:8]
    (dest / f"page{ext}").write_bytes(body)
    detail = f"{filename or media}, {len(body) / 1e3:.0f} kB" + (
        " (truncated)" if truncated else ""
    )
    return f"page{ext}", detail, len(body) < THIN_CHARS


def google_export(url: str) -> list[str]:
    """Direct-download URLs for a Google Docs or Drive link, best first; [] if it is neither."""
    u = urlparse(url)
    m = re.search(r"/d/([\w-]{20,})", u.path)
    if (
        not m or "/d/e/" in u.path
    ):  # published (/d/e/.../pub) docs are ordinary HTML pages
        return []
    doc_id = m.group(1)
    drive = f"https://drive.google.com/uc?export=download&id={doc_id}"
    if u.netloc == "docs.google.com" and u.path.startswith("/document/"):
        return [f"https://docs.google.com/document/d/{doc_id}/export?format=txt", drive]
    if u.netloc == "docs.google.com" and u.path.startswith("/presentation/"):
        return [f"https://docs.google.com/presentation/d/{doc_id}/export/pdf", drive]
    if u.netloc == "docs.google.com" and u.path.startswith("/spreadsheets/"):
        return [
            f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv",
            drive,
        ]
    if u.netloc == "drive.google.com":
        return [drive]
    return []


# ---------------------------------------------------------------- per-target fetch


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return f"URL error: {exc.reason}"
    if isinstance(exc, subprocess.CalledProcessError):
        lines = [
            ln
            for ln in (exc.stderr or "").strip().splitlines()
            if not ln.startswith(("warning:", "hint:"))
        ]
        return f"git {exc.cmd[1]}: {lines[-1] if lines else f'exit {exc.returncode}'}"
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"timed out after {exc.timeout:.0f}s ({' '.join(exc.cmd[:2])})"
    return f"{type(exc).__name__}: {exc}"


def plan_target(url: str, probe: bool) -> tuple[str, GitTarget | None, list[str], str]:
    """(handler, git target, download URLs, folder key) for one link."""
    if target := parse_git(url):
        return "git", target, [url], target.key
    if exports := google_export(url):
        return (
            "google",
            None,
            exports,
            _key("google", re.search(r"/d/([\w-]+)", url).group(1)),
        )
    u = urlparse(url)
    key = _key(u.netloc.removeprefix("www."), u.path.strip("/").replace("/", "_"))[:60]
    if probe and (target := probe_git(url)):
        return "git", target, [url], target.key
    return "http", None, [url], f"{key}_{hashlib.sha1(url.encode()).hexdigest()[:6]}"


def fetch_target(
    handler: str,
    target: GitTarget | None,
    urls: list[str],
    dest: Path,
    max_bytes: int | None,
    refetch: bool,
) -> Result:
    """Fetch one target into dest. Git failures fall back to downloading the linked page."""
    if refetch and dest.exists():
        for child in (
            "repo",
            "LARGE_FILES_SKIPPED.txt",
            *(p.name for p in dest.glob("page*")),
            "links.txt",
        ):
            path = dest / child
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    errors = []
    if target is not None:
        repo_dir = dest / "repo"
        if (repo_dir / ".git").is_dir() and not refetch:
            return Result("cached", handler, "repo/", target.path)
        try:
            return Result(
                "ok", handler, "repo/", target.path, clone(target, dest, max_bytes)
            )
        except Exception as exc:  # noqa: BLE001 - every failure is reported, the run goes on
            shutil.rmtree(repo_dir, ignore_errors=True)
            errors.append(describe_error(exc))
            handler = f"{handler}->http"
    existing = sorted(p.name for p in dest.glob("page*")) if dest.exists() else []
    if existing and not refetch:
        return Result("cached", handler, ", ".join(existing))
    for url in urls:
        try:
            files, detail, thin = save_page(url, dest)
            detail = "; ".join([*errors, detail]) if errors else detail
            return Result("thin" if thin else "ok", handler, files, "", detail)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"{url}: {describe_error(exc)}"
                if len(urls) > 1
                else describe_error(exc)
            )
    return Result("failed", handler, detail="; ".join(errors))


# ---------------------------------------------------------------- reports


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"


INDEX_FIELDS = [
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
]


def write_reports(out: Path, rows: list[dict]) -> None:
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"] in ("failed", "thin")]
    lines = [
        "# Leaderboard links that could not be fetched",
        "",
        f"{sum(r['status'] == 'failed' for r in rows)} failed and {sum(r['status'] == 'thin' for r in rows)} "
        f"thin (under {THIN_CHARS} characters of text: usually a login wall or a JavaScript-only page) "
        f"out of {len(rows)} links.",
        "",
    ]
    if bad:
        lines += ["| user | status | link | error |", "|---|---|---|---|"]
        lines += [
            f"| {r['user']} | {r['status']} | {r['link']} | {r['detail'].replace('|', '/')} |"
            for r in bad
        ]
    (out / "FAILURES.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="leaderboard CSVs (default: latest saved snapshot of each)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV),
        help=f"challenge Space (default: ${URL_ENV})",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download fresh leaderboard snapshots first",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        help="snapshot folder (default: <RESULTS_FOLDER>/leaderboards/<space>)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output folder (default: <RESULTS_FOLDER>/leaderboard_lit/<space>)",
    )
    parser.add_argument(
        "--open-code",
        action="store_true",
        help="only links from rows marked Open Code = Yes",
    )
    parser.add_argument(
        "--user",
        action="append",
        help="only these usernames or their folder names (repeatable, case-insensitive)",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=5,
        help="skip repo files over this size (default 5; 0 = keep all)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="do not try `git ls-remote` on unrecognised URLs",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="fetch again even what is already on disk",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="only list the links and how each would be fetched",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=8, help="parallel downloads (default 8)"
    )
    args = parser.parse_args(argv)

    space = None
    if args.url:
        base = resolve_base_url(args.url)
        space = snapshot_dir(base).name
    if args.csvs:
        paths = args.csvs
    else:
        if not args.url and not args.snapshots:
            parser.error(f"give leaderboard CSVs, --snapshots, --url or set {URL_ENV}")
        folder = args.snapshots or snapshot_dir(base)
        if args.fetch:
            if not args.url:
                parser.error("--fetch needs --url or $" + URL_ENV)
            for lb in find_leaderboards(base):
                try:
                    _dest, note = download(base, lb, folder)
                    print(f"{lb.label}: {note}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{lb.label}: FAILED ({exc})", file=sys.stderr)
        paths = latest_snapshots(folder)
        if not paths:
            raise SystemExit(
                f"No leaderboard snapshots in {folder}; run download-leaderboard or pass --fetch"
            )
    out = args.out or RESULTS_DIR / "leaderboard_lit" / (space or "local")

    links = read_snapshots(paths)
    if args.open_code:
        links = {
            k: v for k, v in links.items() if "yes" in {s.lower() for s in v.open_code}
        }
    if args.user:
        # "Jane Doe" and its folder name "Jane_Doe" both match
        wanted = {slug(u).lower() for u in args.user}
        links = {k: v for k, v in links.items() if slug(v.user).lower() in wanted}
    print(
        f"{len(links)} links from {len(paths)} leaderboards, {len({u for u, _ in links})} users"
    )

    # Group links by (user, target) so several links into one repository share a single clone.
    plans: dict[
        tuple[str, str], tuple[str, GitTarget | None, list[str], list[Link]]
    ] = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        planned = list(
            pool.map(
                lambda link: plan_target(link.url, not args.no_probe), links.values()
            )
        )
    for link, (handler, target, urls, key) in zip(links.values(), planned):
        entry = plans.setdefault((slug(link.user), key), (handler, target, urls, []))
        entry[3].append(link)
    if args.list:
        for (user, key), (handler, target, urls, group) in sorted(
            plans.items(), key=lambda kv: kv[0][0].lower()
        ):
            for link in group:
                print(f"{handler:6}  {user}/{key}  {link.url}")
        return

    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_mb * 1e6) if args.max_mb > 0 else None

    def run(item):
        (user, key), (handler, target, urls, group) = item
        dest = out / user / key
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "source.txt").write_text(
            "".join(f"{link.url}\t{','.join(sorted(link.boards))}\n" for link in group)
        )
        result = fetch_target(handler, target, urls, dest, max_bytes, args.refetch)
        for link in group:
            print(
                f"{result.status:>7}  {link.user}  {link.url}"
                + (
                    f"  ({result.detail})"
                    if result.status in ("failed", "thin")
                    else ""
                ),
                flush=True,
            )
        return item, result

    rows = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for ((user, key), (_, target, _, group)), result in pool.map(
            run, plans.items()
        ):
            for link in group:
                path = target.path if target and target.path else result.path
                rows.append(
                    {
                        "user": link.user,
                        "link": link.url,
                        "status": result.status,
                        "handler": result.handler,
                        "folder": f"{user}/{key}",
                        "fetched": result.fetched,
                        "path": path,
                        "detail": result.detail,
                        "boards": ";".join(sorted(link.boards)),
                        "open_code": ";".join(sorted(link.open_code)),
                    }
                )
    rows.sort(key=lambda r: (r["user"].lower(), r["link"]))
    write_reports(out, rows)
    counts = {
        s: sum(r["status"] == s for r in rows)
        for s in ("ok", "cached", "thin", "failed")
    }
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items() if n) + f" -> {out}")
    if counts["failed"] or counts["thin"]:
        print(f"see {out / 'FAILURES.md'}")


if __name__ == "__main__":
    main()
