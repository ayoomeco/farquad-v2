"""CPU-only boot paths for engine/worker/install.sh.

Fakes PREFIX and stubs git, uv, pip (via uv), and curl. No GPU, no network, no keys.
  (a) older .install-ok-v0.39.1-$TORCH_VER marker + a working venv -> upgrade in place
  (b) exact .install-ok-v0.42.5-$TORCH_VER marker -> exit 0, venv untouched
  (c) venv present, no marker -> fresh path recreates it (uv venv --clear) and exits 0
  (d) older marker but the venv cannot import torch -> fresh path, not upgrade
"""
import os, pathlib, shutil, subprocess, tempfile, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTALL = ROOT / "engine" / "worker" / "install.sh"
TORCH = "2.13.0+cu130"
TV = "0.28.0+cu130"
TA = "2.13.0+cu130"
COMFY_REF = "v0.42.5"
TORCH_VER = "2.13.0"


def fail(msg):
    raise SystemExit(msg)


def script(body):
    # Shebang must be byte 0. A raw triple-quoted block starts with a newline; dedent after that.
    text = textwrap.dedent(body.lstrip("\n"))
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_exe(path, body):
    path.write_text(body)
    path.chmod(0o755)


def make_stubs(bin_dir, site):
    bin_dir.mkdir(parents=True)
    site.mkdir(parents=True)
    (site / "torch.py").write_text(textwrap.dedent(f'''\
        __version__ = "{TORCH}"
        class version:
            cuda = "13.0"
        '''))
    (site / "torchvision.py").write_text(f'__version__ = "{TV}"\n')
    (site / "torchaudio.py").write_text(f'__version__ = "{TA}"\n')
    write_exe(bin_dir / "python-venv", script(r'''
        #!/bin/bash
        # Venv python for the CPU install test. Forwards to system Python with a fake torch.
        # The custom-node recheck is acknowledged here: this machine has no ComfyUI runtime.
        LOG=${INSTALL_STUB_LOG:?}
        SITE=${FAKE_PY_SITE:?}
        export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
        printf 'PY %s\n' "$*" >> "$LOG"
        if [ "${1:-}" = "-" ]; then
          src=$(cat)
          shift || true
          if printf '%s' "$src" | grep -q "required custom nodes"; then
            printf 'PY custom-node-import-check\n' >> "$LOG"
            printf '%s\n' "[install] required custom nodes import OK"
            exit 0
          fi
          printf '%s' "$src" | /usr/bin/python3 - "$@"
          exit $?
        fi
        exec /usr/bin/python3 "$@"
        '''))
    write_exe(bin_dir / "uv", script(r'''
        #!/usr/bin/env python3
        import os, pathlib, shutil, sys
        log = pathlib.Path(os.environ["INSTALL_STUB_LOG"])
        with log.open("a") as f:
            f.write("UV " + " ".join(sys.argv[1:]) + "\n")
        args = sys.argv[1:]
        if args and args[0] == "venv":
            dest = pathlib.Path(args[-1])
            if dest.exists():
                shutil.rmtree(dest)
            bindir = dest / "bin"
            bindir.mkdir(parents=True)
            src = pathlib.Path(os.environ["INSTALL_PY_STUB"])
            target = bindir / "python"
            shutil.copy(src, target)
            target.chmod(0o755)
        sys.exit(0)
        '''))
    write_exe(bin_dir / "git", script(r'''
        #!/usr/bin/env python3
        import os, pathlib, sys
        log = pathlib.Path(os.environ["INSTALL_STUB_LOG"])
        args = sys.argv[1:]
        with log.open("a") as f:
            f.write("GIT " + " ".join(args) + "\n")

        def c_root():
            if "-C" in args:
                return pathlib.Path(args[args.index("-C") + 1])
            return None

        if args and args[0] == "clone":
            dest = pathlib.Path(args[-1])
            if "LatentSync" in dest.name:
                sys.exit(0)
            dest.mkdir(parents=True, exist_ok=True)
            if dest.name == "ComfyUI":
                (dest / ".git").mkdir(exist_ok=True)
                (dest / "comfyui_version.py").write_text('__version__ = "0.42.5"\n')
                req = dest / "requirements.txt"
                if not req.exists():
                    req.write_text("numpy>=1.25.0\ntorch\ntorchvision\ntorchsde\n")
            sys.exit(0)
        root = c_root()
        if root is not None and "checkout" in args and root.name == "ComfyUI":
            root.mkdir(parents=True, exist_ok=True)
            (root / "comfyui_version.py").write_text('__version__ = "0.42.5"\n')
            req = root / "requirements.txt"
            if not req.exists():
                req.write_text("numpy>=1.25.0\ntorch\ntorchvision\ntorchsde\n")
        sys.exit(0)
        '''))
    write_exe(bin_dir / "curl", script(r'''
        #!/usr/bin/env python3
        import io, os, pathlib, sys, tarfile, zipfile
        log = pathlib.Path(os.environ["INSTALL_STUB_LOG"])
        args = sys.argv[1:]
        with log.open("a") as f:
            f.write("CURL " + " ".join(args) + "\n")
        if "-o" not in args:
            sys.exit(0)
        out = pathlib.Path(args[args.index("-o") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.name.endswith(".tgz") or out.name.endswith(".tar.gz"):
            with tarfile.open(out, "w:gz") as tar:
                data = b'__version__ = "0.42.5"\n'
                info = tarfile.TarInfo("ComfyUI-0.42.5/comfyui_version.py")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                req = b"numpy>=1.25.0\ntorch\ntorchvision\ntorchsde\n"
                info = tarfile.TarInfo("ComfyUI-0.42.5/requirements.txt")
                info.size = len(req)
                tar.addfile(info, io.BytesIO(req))
        else:
            with zipfile.ZipFile(out, "w") as z:
                z.writestr("node/__init__.py", "# stub\n")
        sys.exit(0)
        '''))
    write_exe(bin_dir / "sha256sum", script(r'''
        #!/bin/bash
        printf 'SHA %s\n' "$*" >> "${INSTALL_STUB_LOG:?}"
        if [ "${1:-}" = "-c" ] || [ "${2:-}" = "-c" ]; then
          exit 0
        fi
        exec /usr/bin/sha256sum "$@"
        '''))


def working_venv(prefix, py_stub):
    bindir = prefix / "venv" / "bin"
    bindir.mkdir(parents=True)
    target = bindir / "python"
    shutil.copy(py_stub, target)
    target.chmod(0o755)
    canary = prefix / "venv" / "CANARY"
    canary.write_text("keep-this-venv")
    return canary


def baked_comfy(prefix, version):
    comfy = prefix / "ComfyUI"
    (comfy / ".git").mkdir(parents=True)
    (comfy / "comfyui_version.py").write_text(f'__version__ = "{version}"\n')
    (comfy / "requirements.txt").write_text(
        "numpy>=1.25.0\ntorch\ntorchvision\ntorchaudio\ntorchsde\n"
    )
    for name in ("ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite"):
        d = comfy / "custom_nodes" / name
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("# baked node\n")
    return comfy


def run_install(prefix, log_path, stub_bin, site, py_stub):
    log_path.write_text("")
    env = {
        "PATH": str(stub_bin) + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        "PREFIX": str(prefix),
        "INSTALL_STUB_LOG": str(log_path),
        "INSTALL_PY_STUB": str(py_stub),
        "FAKE_PY_SITE": str(site),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        ["/bin/bash", str(INSTALL)],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    return proc, log_path.read_text()


def uv_lines(log):
    return [ln[3:] for ln in log.splitlines() if ln.startswith("UV ")]


def git_lines(log):
    return [ln[4:] for ln in log.splitlines() if ln.startswith("GIT ")]


def flag(cmd, name):
    parts = cmd.split()
    if name not in parts:
        fail(f"{name} missing in: {cmd}")
    return parts[parts.index(name) + 1]


def assert_no_venv_cmd(log):
    for cmd in uv_lines(log):
        if cmd.split()[:1] == ["venv"]:
            fail(f"venv was recreated on the upgrade/exact path: {cmd}")


def main():
    if not INSTALL.is_file():
        fail(f"missing {INSTALL}")
    work = pathlib.Path(tempfile.mkdtemp(prefix="gj-install-"))
    stub_bin = work / "bin"
    site = work / "site"
    make_stubs(stub_bin, site)
    py_stub = stub_bin / "python-venv"
    try:
        case_upgrade(work, stub_bin, site, py_stub)
        case_exact(work, stub_bin, site, py_stub)
        case_fresh_venv(work, stub_bin, site, py_stub)
        case_broken_venv(work, stub_bin, site, py_stub)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("install upgrade tests ok")


def case_upgrade(work, stub_bin, site, py_stub):
    prefix = work / "a-old-marker"
    prefix.mkdir()
    (prefix / f".install-ok-v0.39.1-{TORCH_VER}").write_text("baked\n")
    canary = working_venv(prefix, py_stub)
    inode = (prefix / "venv").stat().st_ino
    baked_comfy(prefix, "0.39.1")
    proc, log = run_install(prefix, work / "a.log", stub_bin, site, py_stub)
    if proc.returncode != 0:
        fail(f"upgrade path failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nLOG:\n{log}")
    out = proc.stdout
    if "path=upgrade" not in out or "v0.39.1" not in out or "reusing venv" not in out or "not recreating" not in out:
        fail(f"upgrade path was not logged:\n{out}")
    if "path=fresh" in out or "path=exact" in out:
        fail(f"upgrade path also took another path:\n{out}")
    if canary.read_text() != "keep-this-venv":
        fail("upgrade recreated the venv (canary gone)")
    if (prefix / "venv").stat().st_ino != inode:
        fail("upgrade replaced the venv directory")
    assert_no_venv_cmd(log)
    if any("torch==2.13.0" in ln and "torch==2.13.0+" not in ln for ln in uv_lines(log)):
        fail(f"upgrade reinstalled torch at TORCH_VER:\n{log}")
    fetches = [ln for ln in git_lines(log) if "fetch" in ln.split()]
    checkouts = [ln for ln in git_lines(log) if "checkout" in ln.split() and COMFY_REF in ln]
    if not fetches or not any(COMFY_REF in ln for ln in fetches):
        fail(f"upgrade did not fetch the tag:\n{log}")
    if not checkouts:
        fail(f"upgrade did not check out {COMFY_REF}:\n{log}")
    if "CURL " in log:
        fail(f"upgrade overlaid instead of checking out the tag:\n{log}")
    ver = (prefix / "ComfyUI" / "comfyui_version.py").read_text()
    if '0.42.5' not in ver:
        fail(f"ComfyUI was not moved to 0.42.5: {ver}")
    pinned = [ln for ln in uv_lines(log) if ln.startswith("pip install") and " -c " in ln and " -r " in ln]
    if len(pinned) != 1:
        fail(f"expected one constrained requirements install, got {pinned!r}\n{log}")
    con = pathlib.Path(flag(pinned[0], "-c")).read_text()
    req = pathlib.Path(flag(pinned[0], "-r")).read_text().splitlines()
    if f"torch=={TORCH}" not in con.splitlines() or f"torchvision=={TV}" not in con.splitlines():
        fail(f"constraints did not pin the baked torch stack:\n{con}")
    if any(ln.strip() in ("torch", "torchvision", "torchaudio") or ln.startswith("torch==") or ln.startswith("torchvision==") for ln in req):
        fail(f"requirements passed to pip still name torch:\n{req}")
    if "torchsde" not in req or "numpy>=1.25.0" not in req:
        fail(f"filtered requirements dropped non-torch lines:\n{req}")
    if f"torch unchanged {TORCH}" not in out or f"torchvision unchanged {TV}" not in out:
        fail(f"upgrade did not confirm torch stayed put:\n{out}")
    if "PY custom-node-import-check" not in log or "required custom nodes import OK" not in out:
        fail(f"upgrade did not recheck custom nodes:\n{out}\n{log}")
    node = prefix / "ComfyUI" / "custom_nodes" / "h3_exact_lora" / "__init__.py"
    if not node.is_file() or "class H3ExactLoraStack" not in node.read_text():
        fail("upgrade did not install the vendored h3_exact_lora node")
    if not (prefix / "post_gjv2.py").is_file():
        fail("upgrade did not copy post_gjv2.py")
    if not (prefix / f".install-ok-{COMFY_REF}-{TORCH_VER}").is_file():
        fail("upgrade did not write the new marker")
    if not (prefix / f".install-ok-v0.39.1-{TORCH_VER}").is_file():
        fail("upgrade removed the old marker")
    if "path=upgrade done" not in out:
        fail(f"upgrade did not finish:\n{out}")


def case_exact(work, stub_bin, site, py_stub):
    prefix = work / "b-exact"
    prefix.mkdir()
    (prefix / f".install-ok-{COMFY_REF}-{TORCH_VER}").write_text("ok\n")
    canary = working_venv(prefix, py_stub)
    inode = (prefix / "venv").stat().st_ino
    proc, log = run_install(prefix, work / "b.log", stub_bin, site, py_stub)
    if proc.returncode != 0:
        fail(f"exact path failed ({proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    if "path=exact" not in proc.stdout or "already installed" not in proc.stdout:
        fail(f"exact path was not logged:\n{proc.stdout}")
    if "path=upgrade" in proc.stdout or "path=fresh" in proc.stdout:
        fail(f"exact marker took another path:\n{proc.stdout}")
    if uv_lines(log) or git_lines(log) or "CURL " in log or "PY " in log:
        fail(f"exact marker ran install commands:\n{log}")
    if canary.read_text() != "keep-this-venv" or (prefix / "venv").stat().st_ino != inode:
        fail("exact path touched the venv")


def case_fresh_venv(work, stub_bin, site, py_stub):
    prefix = work / "c-no-marker"
    prefix.mkdir()
    canary = working_venv(prefix, py_stub)
    proc, log = run_install(prefix, work / "c.log", stub_bin, site, py_stub)
    if proc.returncode != 0:
        fail(f"fresh path failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nLOG:\n{log}")
    out = proc.stdout
    if "path=fresh" not in out:
        fail(f"fresh path was not logged:\n{out}")
    if "path=upgrade" in out:
        fail(f"venv without a marker took the upgrade path:\n{out}")
    clears = [ln for ln in uv_lines(log) if ln.split()[:1] == ["venv"] and "--clear" in ln.split()]
    if len(clears) != 1:
        fail(f"fresh path did not recreate the venv with uv venv --clear:\n{log}")
    if canary.exists():
        fail("fresh path kept the old venv canary; --clear did not recreate it")
    if not (prefix / f".install-ok-{COMFY_REF}-{TORCH_VER}").is_file():
        fail("fresh path did not write the install marker")
    if "path=fresh done" not in out:
        fail(f"fresh path did not finish:\n{out}")
    if not (prefix / "ComfyUI" / "custom_nodes" / "h3_exact_lora" / "__init__.py").is_file():
        fail("fresh path did not install h3_exact_lora")


def case_broken_venv(work, stub_bin, site, py_stub):
    prefix = work / "d-broken"
    prefix.mkdir()
    (prefix / f".install-ok-v0.39.1-{TORCH_VER}").write_text("baked\n")
    bindir = prefix / "venv" / "bin"
    bindir.mkdir(parents=True)
    broken = bindir / "python"
    write_exe(broken, "#!/bin/bash\nexit 1\n")
    (prefix / "venv" / "CANARY").write_text("broken")
    proc, log = run_install(prefix, work / "d.log", stub_bin, site, py_stub)
    if proc.returncode != 0:
        fail(f"broken-venv path failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nLOG:\n{log}")
    out = proc.stdout
    if "path=upgrade" in out:
        fail(f"broken venv was upgraded in place:\n{out}")
    if "not a working venv" not in out or "path=fresh" not in out:
        fail(f"broken venv did not fall through to fresh:\n{out}")
    if not any(ln.split()[:1] == ["venv"] and "--clear" in ln.split() for ln in uv_lines(log)):
        fail(f"broken venv was not recreated:\n{log}")
    if (prefix / "venv" / "CANARY").exists():
        fail("broken venv was reused")
    if not (prefix / f".install-ok-{COMFY_REF}-{TORCH_VER}").is_file():
        fail("broken-venv fresh install did not write the new marker")


if __name__ == "__main__":
    main()
