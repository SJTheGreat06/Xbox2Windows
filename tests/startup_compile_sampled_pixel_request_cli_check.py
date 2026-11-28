"""Offline unit, CLI, warm-reuse, path-guard and mutation suite for tools/startup/compile_sampled_pixel_request.py.

The graphics front end (gfx2hlsl) and the native compiler are replaced by small fakes, so the suite checks what
the tool itself owns: request and companion validation, program slicing, binding comparison, the replay bundle's
identity, warm reuse, damaged-bundle detection and clean exits for every refusal.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STARTUP = ROOT / "tools" / "startup"
TOOL = STARTUP / "compile_sampled_pixel_request.py"
sys.path.insert(0, str(STARTUP))

import compile_sampled_pixel_request as tool  # noqa: E402

gfx = tool.gfx
REAL_RUN = subprocess.run
HEADER = b"HDR!"
PS_BYTES = HEADER + b"PSPSPSPS" + b"tail"
VS_BYTES = HEADER + b"VSVSVSVS" + b"tail"
ROOT_BYTES = b"root-signature"
MANIFEST = "manifest\nud 3\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 startinstance\nsrv 0 tex\nsmp 0 1\n"
BUFFERED_MANIFEST = ("manifest\nud 4\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 startinstance\nudsrc 3 stride 0\n"
                     "srv 0 vb0\nsrv 1 vb1\n")
SAMPLED_STDOUT = "srv 0 tex\nsampler 0 1\n"
BUFFERED_STDOUT = "srv 0 vb0\n"
TABLES = [{"type": 0, "vis": 0}] * 2


class FakePipeline:
    def __init__(self, vs, ps, root, sampler_heap=False, rtv_formats=()):
        self.vs, self.ps, self.root = vs, ps, root
        self.params = FakePipeline.params
        self.flat, self.notes, self.lod_unvalidated = [], set(), set()

    params = [0]
    manifest_text = MANIFEST

    def manifest(self):
        return self.manifest_text

    def hlsl_ps(self):
        return "// synthetic pixel shader\n"


def default_decode(raw):
    return [(0, 4, "s_nop", None), (4, 4, "s_endpgm", None)]


@contextmanager
def fake_gfx(*, decode=default_decode, buffered=False, stages=None, root=ROOT_BYTES, root_layout=None,
             unsupported=False, manifest=None):
    """Patch the parts of gfx2hlsl that need real machine code; keep the real key and hash functions."""
    stages = {"vs": VS_BYTES, "ps": PS_BYTES} if stages is None else stages
    FakePipeline.params = [0, 1] if buffered else [0]
    FakePipeline.manifest_text = manifest or (BUFFERED_MANIFEST if buffered else MANIFEST)

    def load_request(folder, variant):
        if unsupported:
            raise gfx.Unsupported("synthetic unsupported request")
        return stages, root, ("R11G11B10_FLOAT", "R16_FLOAT"), "rtv 26 54", None, None, None

    layout = {"flags": 1, "params": TABLES} if root_layout is None else root_layout
    with (mock.patch.object(gfx, "load_request", side_effect=load_request),
          mock.patch.object(gfx, "parse_root", return_value=layout),
          mock.patch.object(gfx, "Pipeline", FakePipeline),
          mock.patch.object(gfx, "strip_header", side_effect=lambda blob: blob[len(HEADER):]),
          mock.patch.object(gfx, "decode", side_effect=decode)):
        yield


def fake_compiler(stdout=SAMPLED_STDOUT, calls=None):
    def run(command, check=False, **kwargs):
        assert check is True and kwargs.get("text") is True
        if calls is not None:
            calls.append([str(item) for item in command])
        compiler, sdk, cache, jobs, count, batch, vs, vs_cso, ps, ps_cso = command
        assert Path(compiler).is_file() and Path(sdk).is_dir() and jobs == "--jobs" and count == "2"
        assert Path(vs).is_file() and Path(ps).is_file()
        Path(vs_cso).write_bytes(b"DXBC-VS")
        Path(ps_cso).write_bytes(b"DXBC-PS")
        return subprocess.CompletedProcess(command, 0, stdout=stdout)
    return run


def make_request(work, *, buffered=False, companions=(".a", ".b", ".root")):
    """req_<base>.a/.b/.root and the req_<variant>.rtv marker; returns (base, marker)."""
    base = gfx.pipeline_key(VS_BYTES, PS_BYTES, ROOT_BYTES)
    contents = {".a": VS_BYTES, ".b": PS_BYTES, ".root": ROOT_BYTES}
    for suffix in companions:
        (work / f"req_{base:016x}{suffix}").write_bytes(contents[suffix])
    request = f"rtv1 {base} {'10' if buffered else '26 54'}".encode()
    marker = work / f"req_{gfx.fnv(request):016x}.rtv"
    marker.write_bytes(request)
    return base, marker


def environment(root):
    compiler = root / "fake_compile_dxil.exe"
    compiler.write_bytes(b"FAKE")
    sdk = root / "sdk"
    sdk.mkdir()
    return compiler, sdk, root / "bundles", root / "cache"


def arguments(marker, output, compiler, sdk, cache, *extra):
    return [str(marker), str(output), "--compiler", str(compiler), "--sdk", str(sdk), "--cache", str(cache), *extra]


def run_main(argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = tool.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def run_cli(*args):
    return REAL_RUN([sys.executable, "-B", str(TOOL), *map(str, args)], capture_output=True, text=True, timeout=60)


class Fixture(unittest.TestCase):
    def setUp(self):
        self._folder = TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.root = Path(self._folder.name).resolve()
        self.compiler, self.sdk, self.output, self.cache = environment(self.root)
        self.work = self.root / "request"
        self.work.mkdir()

    def argv(self, marker, *extra):
        return arguments(marker, self.output, self.compiler, self.sdk, self.cache, *extra)

    def assert_refused(self, argv, fragment, *, cli=False, fake=None, **gfx_options):
        """main exits 1 with the diagnostic on stderr only, and no traceback."""
        with fake_gfx(**gfx_options), mock.patch.object(tool.subprocess, "run", side_effect=fake or fake_compiler()):
            code, out, err = run_main(argv)
        self.assertEqual(code, 1, (out, err))
        self.assertNotIn("Traceback", err)
        self.assertIn(fragment, err)
        if cli:
            done = run_cli(*argv)
            self.assertEqual(done.returncode, 1, done.stderr)
            self.assertNotIn("Traceback", done.stderr)
            self.assertIn(fragment, done.stderr)


class PositiveTests(Fixture):
    def test_default_replay_warm_reuse_and_damaged_bundle(self):
        _, marker = make_request(self.work)
        calls = []
        with fake_gfx(), mock.patch.object(tool.subprocess, "run", side_effect=fake_compiler(calls=calls)):
            code, out, err = run_main(self.argv(marker))
            self.assertEqual((code, err), (0, ""))
            bundle = Path(out.strip().splitlines()[-1])
            self.assertEqual(list(self.output.iterdir()), [bundle])
            meta = json.loads((bundle / "replay.json").read_bytes())
            self.assertEqual(meta["profile"], "gfx1013-rgb11-r16-ps")
            self.assertEqual((meta["pixel_offset"], meta["pixel_bytes"], meta["vertex_bytes"]), (4, 8, 8))
            self.assertEqual(meta["rtv_formats"], ["R11G11B10_FLOAT", "R16_FLOAT"])
            for name, digest in meta["sha256"].items():
                self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), digest, name)
            self.assertEqual((bundle / next(n for n in meta["sha256"] if n.endswith(".ps.bin"))).read_bytes(),
                             PS_BYTES[4:12])
            self.assertEqual(calls[0][3:6], ["--jobs", "2", "gfx1013-sampled-pair"])

            # A second run with identical inputs reuses the published bundle and leaves no temporary folder.
            again = run_main(self.argv(marker))
            self.assertEqual(again, (0, out, ""))
            self.assertEqual(list(self.output.iterdir()), [bundle])
            self.assertEqual(len(calls), 2)

            # build() still returns the bundle path for library callers (including string compiler/sdk args).
            with redirect_stdout(io.StringIO()):
                self.assertEqual(tool.build(marker, self.output, str(self.compiler), str(self.sdk), self.cache), bundle)

            # Damage a published artifact: the next run refuses with the artifact's name.
            vs_cso = next(bundle.glob("*.vs.cso"))
            saved_cso = vs_cso.read_bytes()
            vs_cso.write_bytes(b"damaged")
            code, out, err = run_main(self.argv(marker))
            self.assertEqual((code, out.strip()), (1, SAMPLED_STDOUT.strip()))
            self.assertIn("damaged replay bundle: ", err)
            self.assertNotIn("Traceback", err)

            # Missing published artifact or tampered replay.json also refuses cleanly on collision.
            vs_cso.unlink()
            code, _, err = run_main(self.argv(marker))
            self.assertEqual(code, 1)
            self.assertIn("damaged replay bundle: ", err)
            self.assertNotIn("Traceback", err)
            vs_cso.write_bytes(saved_cso)
            (bundle / "replay.json").write_bytes(b"{}")
            code, _, err = run_main(self.argv(marker))
        self.assertEqual(code, 1)
        self.assertIn("damaged replay bundle: replay.json", err)
        self.assertNotIn("Traceback", err)

    def test_buffered_vertex_replay(self):
        _, marker = make_request(self.work, buffered=True)
        calls = []
        with fake_gfx(buffered=True), mock.patch.object(
                tool.subprocess, "run", side_effect=fake_compiler(BUFFERED_STDOUT, calls)):
            code, out, err = run_main(self.argv(marker, "--buffered-vertex"))
        self.assertEqual((code, err), (0, ""))
        bundle = Path(out.strip().splitlines()[-1])
        meta = json.loads((bundle / "replay.json").read_bytes())
        self.assertEqual((meta["profile"], meta["vertex_profile"]), ("ps_6_0", "gfx1013-buffered-vs"))
        self.assertEqual(meta["entry"]["pixel_frontend"], "gfx2hlsl")
        self.assertEqual(calls[0][5], "gfx1013-buffered-pair")
        self.assertTrue(next(bundle.glob("*.ps.hlsl")).is_file())
        self.assertTrue(calls[0][8].endswith(".ps.hlsl"))

    def test_argparse_errors_exit_two(self):
        for argv, fragment in (
            ([], "required"),
            (["only"], "required"),
            (["a", "b", "--compiler", "c", "--sdk", "s", "--cache", "k", "--nope"], "unrecognized"),
        ):
            with self.subTest(argv=argv):
                code, out, err = run_main(argv)
                self.assertEqual((code, out), (2, ""))
                self.assertIn(fragment, err)
                self.assertNotIn("Traceback", err)


class RefusalTests(Fixture):
    def test_missing_and_hostile_paths(self):
        _, marker = make_request(self.work)
        missing = self.root / "missing"
        self.assert_refused(self.argv(self.work / "req_0000000000000000.rtv"), "marker is not an existing file",
                            cli=True)
        self.assert_refused(self.argv(self.work), "marker is not an existing file")
        self.assert_refused(arguments(marker, self.output, missing, self.sdk, self.cache),
                            "compiler is not an existing file")
        self.assert_refused(arguments(marker, self.output, self.compiler, self.compiler, self.cache),
                            "sdk is not an existing directory")
        self.assert_refused(arguments(marker, self.output, self.compiler, missing, self.cache),
                            "sdk is not an existing directory", cli=True)

        regular = self.root / "regular_file"
        regular.write_bytes(b"keep")
        self.assert_refused(arguments(marker, regular, self.compiler, self.sdk, self.cache),
                            "output path is an existing regular file", cli=True)
        self.assert_refused(arguments(marker, self.output, self.compiler, self.sdk, regular),
                            "cache path is an existing regular file")
        self.assertEqual(regular.read_bytes(), b"keep")
        self.assertFalse(self.output.exists())

        # An output below a regular file cannot be created: a clean refusal, not a traceback.
        self.assert_refused(arguments(marker, regular / "child", self.compiler, self.sdk, self.cache), "")
        self.assertEqual(regular.read_bytes(), b"keep")

        target = self.root / "real_target"
        target.mkdir()
        link = self.root / "linked_output"
        cache_link = self.root / "linked_cache"
        marker_link = self.work / ("req_" + "a" * 16 + ".rtv")
        try:
            link.symlink_to(target, target_is_directory=True)
            cache_link.symlink_to(target, target_is_directory=True)
            marker_link.symlink_to(marker)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314:
                raise
            for name, argv_pair, fragment in (
                ("linked_output", arguments(marker, link, self.compiler, self.sdk, self.cache),
                 "output path is a symbolic link"),
                ("linked_cache", arguments(marker, self.output, self.compiler, self.sdk, cache_link),
                 "cache path is a symbolic link"),
                (marker_link.name, self.argv(marker_link), "marker is not an existing file"),
            ):
                patch = mock.patch.object(Path, "is_symlink", autospec=True,
                                          side_effect=lambda path, target_name=name: path.name == target_name)
                with patch:
                    self.assert_refused(argv_pair, fragment)
        else:
            self.assert_refused(arguments(marker, link, self.compiler, self.sdk, self.cache),
                                "output path is a symbolic link", cli=True)
            self.assert_refused(arguments(marker, self.output, self.compiler, self.sdk, cache_link),
                                "cache path is a symbolic link", cli=True)
            self.assert_refused(self.argv(marker_link), "marker is not an existing file", cli=True)
        self.assertEqual(list(target.iterdir()), [])

    def test_marker_request_and_companion_refusals(self):
        base, marker = make_request(self.work)
        stem = f"req_{base:016x}"

        bad_name = self.work / "marker.rtv"
        bad_name.write_bytes(marker.read_bytes())
        self.assert_refused(self.argv(bad_name), "expected a req_<16 hex digits>.rtv marker")

        # The request names the other format pair, or does not hash to its own marker name.
        self.assert_refused(self.argv(marker, "--buffered-vertex"), "expected a matching RGBA16_FLOAT request")
        forged = self.work / f"req_{1:016x}.rtv"
        forged.write_bytes(marker.read_bytes())
        self.assert_refused(self.argv(forged), "expected a matching R11G11B10_FLOAT / R16_FLOAT request")

        for suffix in (".a", ".b", ".root"):
            companion = self.work / (stem + suffix)
            saved = companion.read_bytes()
            companion.unlink()
            self.assert_refused(self.argv(marker), f"missing or non-regular companion file {companion.name}")
            companion.mkdir()
            self.assert_refused(self.argv(marker), f"missing or non-regular companion file {companion.name}")
            self.assertFalse(self.output.exists(), suffix)
            companion.rmdir()
            companion.write_bytes(saved)

        optional = self.work / (stem + ".ia")
        optional.mkdir()
        self.assert_refused(self.argv(marker), "missing or non-regular companion file " + optional.name)
        optional.rmdir()

        (self.work / (stem + ".b")).write_bytes(VS_BYTES)
        self.assert_refused(self.argv(marker), "graphics stage, root or input layout key mismatch")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_graphics_profile_and_program_refusals(self):
        _, marker = make_request(self.work)
        argv = self.argv(marker)
        self.assert_refused(argv, "unexpected graphics root layout", root_layout={"flags": 1, "params": TABLES[:1]})
        self.assert_refused(argv, "missing program boundary",
                            decode=lambda raw: [(0, 4, "s_nop", None), (4, 4, "s_nop", None)])
        self.assert_refused(argv, "missing program boundary", decode=lambda raw: [])
        self.assert_refused(argv, "invalid program span", decode=lambda raw: [(len(raw), 4, "s_endpgm", None)])
        self.assert_refused(argv, "invalid program span", decode=lambda raw: [(0, 0, "s_endpgm", None)])
        self.assert_refused(argv, "invalid program span", decode=lambda raw: [(-4, 4, "s_endpgm", None)])
        self.assert_refused(argv, "empty graphics program", stages={"vs": VS_BYTES, "ps": b""})
        self.assert_refused(argv, "empty graphics program", stages={"vs": VS_BYTES, "ps": HEADER})
        self.assert_refused(argv, "unexpected vertex entry constants",
                            manifest="manifest\nud 3\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 other\n")
        self.assert_refused(argv, "synthetic unsupported request", unsupported=True)

    def test_compiler_and_binding_refusals(self):
        _, marker = make_request(self.work)
        argv = self.argv(marker)
        self.assert_refused(argv, "native pixel bindings disagree with paired graphics manifest",
                            fake=fake_compiler("srv 0 tex\nsampler 0 2\n"))
        self.assert_refused(argv, "native pixel bindings disagree with paired graphics manifest",
                            fake=fake_compiler(""))
        failure = subprocess.CalledProcessError(3, ["compile_dxil"])
        self.assert_refused(argv, "returned non-zero exit status 3", fake=mock.Mock(side_effect=failure))
        self.assert_refused(argv, "simulated launch failure", fake=mock.Mock(side_effect=OSError("simulated launch failure")))

        _, buffered = make_request(self.root / "request", buffered=True)
        self.assert_refused(self.argv(buffered, "--buffered-vertex"),
                            "native vertex buffer registers disagree with paired graphics manifest",
                            fake=fake_compiler("srv 0 other\n"), buffered=True)
        self.assert_refused(self.argv(buffered, "--buffered-vertex"),
                            "native vertex buffer registers disagree with paired graphics manifest",
                            fake=fake_compiler(""), buffered=True)


class MutationTests(unittest.TestCase):
    def test_fast_self_mutations_are_killed(self):
        def survives(case, method):
            result = unittest.TestResult()
            case(method).run(result)
            return result.wasSuccessful()

        unguarded = tool.write_program
        original_bindings = tool.bindings

        def old_write_program(path, container):
            raw = gfx.strip_header(container)
            instructions = gfx.decode(raw)
            if not instructions or instructions[-1][2] != "s_endpgm":
                raise ValueError("missing program boundary")
            end, size, _, _ = instructions[-1]
            path.write_bytes(raw[:end + size])
            return len(container) - len(raw)

        fake_hash = types.SimpleNamespace(
            sha256=lambda data=b"": types.SimpleNamespace(hexdigest=lambda: "0" * 64))
        mutants = (
            ("no_path_validation", mock.patch.object(tool, "_validate_paths", lambda *args: None),
             RefusalTests, "test_missing_and_hostile_paths"),
            ("unguarded_write_program", mock.patch.object(tool, "write_program", old_write_program),
             RefusalTests, "test_graphics_profile_and_program_refusals"),
            ("ignore_binding_mismatch", mock.patch.object(tool, "bindings", lambda lines, tags: ([], [])),
             RefusalTests, "test_compiler_and_binding_refusals"),
            ("skip_damage_detection", mock.patch.object(tool, "hashlib", fake_hash),
             PositiveTests, "test_default_replay_warm_reuse_and_damaged_bundle"),
            ("skip_companion_check", mock.patch.object(Path, "is_file", autospec=True, return_value=True),
             RefusalTests, "test_marker_request_and_companion_refusals"),
        )
        self.assertIs(tool.write_program, unguarded)
        self.assertIs(tool.bindings, original_bindings)
        for label, patcher, case, method in mutants:
            with patcher:
                self.assertFalse(survives(case, method), f"mutant {label} survived")


if __name__ == "__main__":
    unittest.main()
