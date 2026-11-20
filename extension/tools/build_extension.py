"""
Validate the extension, optionally bump its version and pack it for distribution.

    python3 extension/tools/build_extension.py              # validate + zip the current version
    python3 extension/tools/build_extension.py patch        # 1.0.0 -> 1.0.1, then zip
    python3 extension/tools/build_extension.py minor|major
    python3 extension/tools/build_extension.py --check      # validate only

Output: extension/dist/captchosa-extension-<version>.zip (dist/ is git-ignored).
Only runtime files go into the zip (no tests, tools, docs or icon sources).
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent
DIST = EXT / "dist"
EXCLUDE_DIRS = {"test", "tools", "docs", "dist", "src", "__pycache__", "node_modules"}
EXCLUDE_FILES = {"README.md", "CHANGELOG.md", ".gitignore"}
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
FORBIDDEN = [re.compile(p) for p in (r"\beval\s*\(", r"new\s+Function\s*\(", r"importScripts\s*\(\s*['\"]https?:",
                                     r"<script[^>]+src=['\"]https?:")]


class BuildError(Exception):
    pass


def load_manifest():
    try:
        return json.loads((EXT / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise BuildError(f"manifest.json: {exc}") from exc


def referenced_files(m):
    files = [m["background"]["service_worker"], m["action"]["default_popup"].split("?")[0],
             m["options_ui"]["page"].split("?")[0]]
    files += list(m.get("icons", {}).values()) + list(m["action"].get("default_icon", {}).values())
    for cs in m.get("content_scripts", []):
        files += cs.get("js", []) + cs.get("css", [])
    return files


def validate(m):
    errors = []
    if m.get("manifest_version") != 3:
        errors.append("manifest_version must be 3")
    if not VERSION_RE.match(str(m.get("version", ""))):
        errors.append(f"version {m.get('version')!r} is not X.Y.Z")
    for f in referenced_files(m):
        if not (EXT / f).is_file():
            errors.append(f"missing file: {f}")
    default = m.get("default_locale")
    locales = sorted(p.name for p in (EXT / "_locales").iterdir() if p.is_dir())
    if default not in locales:
        errors.append(f"default_locale {default!r} has no _locales folder")
    base = set(json.loads((EXT / "_locales" / default / "messages.json").read_text()))
    for lang in locales:
        keys = set(json.loads((EXT / "_locales" / lang / "messages.json").read_text()))
        if keys != base:
            errors.append(f"_locales/{lang}: missing {sorted(base - keys)} extra {sorted(keys - base)}")
    used = set()
    for path in EXT.rglob("*"):
        if path.suffix in (".js", ".html", ".json") and not EXCLUDE_DIRS & set(path.relative_to(EXT).parts):
            text = path.read_text(encoding="utf-8")
            used |= set(re.findall(r"__MSG_(\w+)__", text))
            used |= set(re.findall(r"data-i18n(?:-[a-z]+)?=\"(\w+)\"", text))
            if path.suffix in (".js", ".html"):
                for pat in FORBIDDEN:
                    if pat.search(text):
                        errors.append(f"{path.relative_to(EXT)}: forbidden pattern {pat.pattern}")
    for key in sorted(used - base - {"@@ui_locale"}):
        errors.append(f"message {key!r} is used but not defined")
    if errors:
        raise BuildError("\n  ".join(["validation failed:"] + errors))
    return locales


def bump(version, part):
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_version(m, version):
    m["version"] = version
    m["version_name"] = version
    (EXT / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")


def pack(version):
    DIST.mkdir(exist_ok=True)
    out = DIST / f"captchosa-extension-{version}.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(EXT.rglob("*")):
            rel = path.relative_to(EXT)
            if path.is_dir() or EXCLUDE_DIRS & set(rel.parts) or rel.name in EXCLUDE_FILES:
                continue
            if rel.suffix in (".py", ".pyc", ".log"):
                continue
            z.write(path, rel.as_posix())
            count += 1
    return out, count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bump", nargs="?", choices=("patch", "minor", "major"))
    ap.add_argument("--check", action="store_true", help="validate only")
    args = ap.parse_args()
    try:
        m = load_manifest()
        locales = validate(m)
        print(f"manifest ok: version {m['version']}, locales {', '.join(locales)}")
        if args.check:
            return 0
        if args.bump:
            new = bump(m["version"], args.bump)
            write_version(m, new)
            print(f"version bumped to {new} (add a CHANGELOG.md entry)")
        out, count = pack(m["version"])
        print(f"wrote {out} ({count} files)")
        return 0
    except BuildError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
