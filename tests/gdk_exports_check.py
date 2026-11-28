#!/usr/bin/env python3
"""Checks for tools/gdk_exports.py and tools/gdk_shim_coverage.py.

Uses synthetic dumpbin text and synthetic .def files. It needs neither Windows
nor the GDK; the PE scanner is a thin wrapper over pefile and is exercised by
the real inventory outside the repository.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_exports as exports  # noqa: E402
import gdk_shim_coverage as coverage  # noqa: E402


DLL_TEXT = """Microsoft (R) COFF/PE Dumper Version 14.44.35207.0

Dump of file xg_xs.dll

File Type: DLL

  Exports

    ordinal hint RVA      name

          1    0 00001000 XGComputeBufferLayout
          2    1 00001200 XGComputeTexture2DLayout
          3    2 00001400 IID_IXGThing
"""

LIB_TEXT = """Dump of file xmem.lib

File Type: LIBRARY

Exports

       ordinal    name

              1 XMemAlloc
              XMemFree
              ?XMemInternal@@YAXXZ
"""


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def check_parse():
    dll = exports.parse_dumpbin_text(DLL_TEXT)
    check(dll == [
        {"ordinal": 1, "name": "XGComputeBufferLayout", "rva": "0x1000"},
        {"ordinal": 2, "name": "XGComputeTexture2DLayout", "rva": "0x1200"},
        {"ordinal": 3, "name": "IID_IXGThing", "rva": "0x1400"},
    ], f"DLL parse {dll}")
    lib = exports.parse_dumpbin_text(LIB_TEXT)
    check(lib == [
        {"ordinal": 1, "name": "XMemAlloc", "rva": None},
        {"ordinal": None, "name": "XMemFree", "rva": None},
        {"ordinal": None, "name": "?XMemInternal@@YAXXZ", "rva": None},
    ], f"LIB parse {lib}")
    for marker in ("Summary", "Dump of file another.lib"):
        capture = (
            "Exports\nordinal name\n1 BeforeMarker\n"
            + marker
            + "\n2 AfterMarker\n"
        )
        parsed = exports.parse_dumpbin_text(capture)
        check(
            [item["name"] for item in parsed] == ["BeforeMarker"],
            f"parser continued past {marker}: {parsed}",
        )

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "xg_xs.txt"
        source.write_text(DLL_TEXT, encoding="utf-8")
        output = directory / "exports.json"
        code = exports.main([
            "parse-dumpbin", str(source), "--json", str(output), "--kind", "dll", "--sdk", "999"
        ])
        check(code == 0, f"parse-dumpbin exit {code}")
        document = json.loads(output.read_text(encoding="utf-8"))
        check(document["count"] == 3, f"count {document['count']}")
        check(document["sdk"] == "999", f"sdk {document['sdk']}")
        check(document["modules"][0]["module"] == "xg_xs", f"module {document['modules'][0]['module']}")
        check(document["modules"][0]["sdk"] == "999", f"module sdk {document['modules'][0]['sdk']}")
        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest().upper()
        check(document["modules"][0]["sha256"] == expected_hash, "capture sha256")
        lib_capture = directory / "d3d12_xs.lib.txt"
        lib_capture.write_text(LIB_TEXT, encoding="utf-8")
        collected = exports.collect_dumpbin(lib_capture, "lib")
        check(collected[0]["module"] == "d3d12_xs", f"module suffix {collected[0]['module']}")
    print("exports: DLL, LIB, ordinal and CLI parsing passed")


def check_coverage():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        inventory = {
            "sdk": "261000",
            "modules": [
                {
                    "file": "xg_xs.dll",
                    "exports": [
                        {"ordinal": 1, "name": "XGComputeBufferLayout", "rva": "0x1000"},
                        {"ordinal": 2, "name": "XGComputeTexture2DLayout", "rva": "0x1200"},
                        {"ordinal": 3, "name": "IID_IXGThing", "rva": "0x1400"},
                    ],
                },
                {
                    "file": "xmem.lib",
                    "exports": [
                        {"ordinal": None, "name": "XMemAlloc", "rva": None},
                        {"ordinal": None, "name": "XMemFree", "rva": None},
                    ],
                },
            ],
        }
        inventory_path = directory / "exports.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        xg_def = directory / "xg_xs.def"
        xg_def.write_text(
            "LIBRARY xg_xs\nEXPORTS\nXGComputeBufferLayout\nIID_IXGThing DATA\nx2w_local\n",
            encoding="utf-8",
        )
        xmem_def = directory / "xmem.def"
        xmem_def.write_text(
            "LIBRARY xmem\nEXPORTS\nXMemAlloc @1\nXMemFree\n?XMemInternal@@YAXXZ\n",
            encoding="utf-8",
        )
        output_json = directory / "coverage.json"
        output_md = directory / "coverage.md"
        code = coverage.main([
            "--exports", str(inventory_path),
            "--def", "xg_xs=" + str(xg_def),
            "--def", "xmem=" + str(xmem_def),
            "--json", str(output_json),
            "--markdown", str(output_md),
        ])
        check(code == 0, f"coverage exit {code}")
        rows = {row["module"]: row for row in json.loads(output_json.read_text(encoding="utf-8"))}
        xg = rows["xg_xs"]
        check(xg["gdk_functions"] == 2, f"xg functions {xg['gdk_functions']}")
        check(xg["shim_functions"] == 2, f"xg shim functions {xg['shim_functions']}")
        check(xg["covered_functions"] == 1, f"xg covered {xg['covered_functions']}")
        check(xg["missing_functions"] == ["XGComputeTexture2DLayout"], f"xg missing {xg['missing_functions']}")
        check(xg["covered_data"] == 1, f"xg data {xg['covered_data']}")
        check(xg["extra"] == ["x2w_local"], f"xg extra {xg['extra']}")
        check(xg["ordinals"] == {
            "IID_IXGThing": 3, "XGComputeBufferLayout": 1, "XGComputeTexture2DLayout": 2
        }, f"xg ordinals {xg['ordinals']}")
        check(rows["xmem"]["def_ordinals"] == {"XMemAlloc": 1}, f"xmem def ordinals {rows['xmem']['def_ordinals']}")
        check("?XMemInternal@@YAXXZ" in rows["xmem"]["extra"], f"decorated name {rows['xmem']['extra']}")
        check("| xg_xs | 2 | 2 | 1 | 1 |" in output_md.read_text(encoding="utf-8"), "markdown table")
        check(coverage.main([
            "--exports", str(inventory_path),
            "--def", "xg_xs=" + str(xg_def),
            "--fail-on-missing",
        ]) == 1, "fail-on-missing exit")
    print("coverage: functions, data, extras, ordinals and exit codes passed")


def check_merge():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        inventory = {
            "modules": [
                {
                    "module": "d3d12_xs",
                    "file": "d3d12_xs.lib.txt",
                    "exports": [
                        {"ordinal": 1, "name": "D3D12XLibOnly", "rva": None},
                        {"ordinal": None, "name": "Shared", "rva": None},
                    ],
                },
                {
                    "module": "d3d12_xs",
                    "file": "d3d12_xs.dll.txt",
                    "exports": [
                        {"ordinal": 2, "name": "Shared", "rva": "0x1000"},
                        {"ordinal": 3, "name": "D3D12XDllOnly", "rva": "0x2000"},
                    ],
                },
            ],
        }
        merged = coverage.load_exports(inventory)
        check(set(merged) == {"d3d12_xs"}, f"merged keys {set(merged)}")
        check(
            {item["name"] for item in merged["d3d12_xs"]["exports"]}
            == {"D3D12XLibOnly", "Shared", "D3D12XDllOnly"},
            f"merged exports {merged['d3d12_xs']['exports']}",
        )
        fallback = coverage.load_exports([
            {"file": "d3d12_xs.lib.txt", "exports": [{"name": "FromLib"}]}
        ])
        check(set(fallback) == {"d3d12_xs"}, f"fallback module key {set(fallback)}")
        inventory_path = directory / "exports.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        def_path = directory / "d3d12_xs.def"
        def_path.write_text(
            "LIBRARY d3d12_xs\nEXPORTS\nShared\nD3D12XLibOnly\n", encoding="utf-8"
        )
        output = directory / "coverage.json"
        check(coverage.main([
            "--exports", str(inventory_path),
            "--def", "d3d12_xs=" + str(def_path),
            "--json", str(output),
        ]) == 0, "merge coverage exit")
        row = json.loads(output.read_text(encoding="utf-8"))[0]
        check(row["gdk_functions"] == 3, f"merge functions {row['gdk_functions']}")
        check(row["covered_functions"] == 2, f"merge covered {row['covered_functions']}")
        check(row["missing_functions"] == ["D3D12XDllOnly"], f"merge missing {row['missing_functions']}")
    print("exports: LIB+DLL merge and module-name key passed")


def check_unmeasured():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        empty_capture = directory / "empty.txt"
        empty_capture.write_text("not a dumpbin export capture\n", encoding="utf-8")
        collected = exports.collect_dumpbin(empty_capture, "dll")
        check(collected[0].get("error") == "no exports parsed", f"empty capture {collected}")

        partial_capture = directory / "partial.txt"
        partial_capture.write_text(
            "Exports\nordinal hint RVA name\n"
            "1 0 00001000 ParsedExport\n"
            "this row cannot be parsed\n",
            encoding="utf-8",
        )
        partial = exports.collect_dumpbin(partial_capture, "dll")[0]
        check(partial["exports"] == [
            {"ordinal": 1, "name": "ParsedExport", "rva": "0x1000"}
        ], f"partial exports {partial['exports']}")
        check(partial["error"] == "1 unparsed export line(s)", f"partial error {partial}")

        inventory = {
            "modules": [
                collected[0],
                partial,
            ]
        }
        inventory_path = directory / "exports.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        empty_def = directory / "empty.def"
        empty_def.write_text("LIBRARY empty\nEXPORTS\nAnything\n", encoding="utf-8")
        partial_def = directory / "partial.def"
        partial_def.write_text("LIBRARY partial\nEXPORTS\nParsedExport\n", encoding="utf-8")
        absent_def = directory / "absent.def"
        absent_def.write_text("LIBRARY absent\nEXPORTS\nAnything\n", encoding="utf-8")
        output = directory / "coverage.json"
        markdown = directory / "coverage.md"
        check(coverage.main([
            "--exports", str(inventory_path),
            "--def", "empty=" + str(empty_def),
            "--def", "partial=" + str(partial_def),
            "--def", "absent=" + str(absent_def),
            "--json", str(output),
            "--markdown", str(markdown),
            "--fail-on-missing",
        ]) == 1, "unmeasured fail-on-missing exit")
        rows = {
            row["module"]: row
            for row in json.loads(output.read_text(encoding="utf-8"))
        }
        check(rows["empty"]["unmeasured"] == "no exports parsed", f"empty row {rows['empty']}")
        check(
            rows["partial"]["unmeasured"] == "1 unparsed export line(s)",
            f"partial row {rows['partial']}",
        )
        check(rows["absent"]["unmeasured"] == "no inventory entry", f"absent row {rows['absent']}")
        check("Unmeasured: no inventory entry" in markdown.read_text(encoding="utf-8"), "markdown status")
    print("exports: empty, absent and fail-on-missing rows passed")


def main():
    check_parse()
    check_coverage()
    check_merge()
    check_unmeasured()
    print("gdk_exports_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
