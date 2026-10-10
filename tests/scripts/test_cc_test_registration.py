# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise native test registration without compiling or running test bodies."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CMAKE = shutil.which("cmake")
_CTEST = shutil.which("ctest")
_NINJA = shutil.which("ninja")

_CASES = (
    ("TEST", "PlainSuite", "PlainCase"),
    ("TEST_F", "FixtureSuite", "FixtureCase"),
    ("TEST_P", "ParameterSuite", "ParameterCase"),
    ("TYPED_TEST", "TypedSuite", "TypedCase"),
    ("TYPED_TEST_P", "TypedParameterSuite", "TypedParameterCase"),
    ("TEST", "DISABLED_DisabledSuite", "PlainCase"),
    ("TEST", "DisabledCaseSuite", "DISABLED_PlainCase"),
    ("TEST_F", "DISABLED_DisabledFixture", "DISABLED_FixtureCase"),
)
_LITERALS = r'${source_variable};@source_variable@;"quotes";[=[brackets]=];$<TARGET_FILE:not_a_target>;backslash\\value'

_PROJECT = r"""
cmake_minimum_required(VERSION 3.26)
project(CcTestRegistration LANGUAGES NONE)
set(BUILD_TESTING ON)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "${XLLM_CC_TEST_MODULE_DIR}")
include(cc_test)

# A compiler is never enabled. Fail if a caller accidentally builds a binary.
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_COMMAND> -E false")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_COMMAND> -E false")
add_custom_target(all_tests)
add_library(fixture_dependency INTERFACE)

set(_configure_count 0)
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/configure-count.txt")
  file(READ "${CMAKE_CURRENT_BINARY_DIR}/configure-count.txt" _configure_count)
endif()
math(EXPR _configure_count "${_configure_count} + 1")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/configure-count.txt" "${_configure_count}")

cc_test(
  NAME registration_test
  SRCS a/same_test.cpp "${CMAKE_CURRENT_SOURCE_DIR}/b/same_test.cpp"
  DEPS fixture_dependency
  ARGS "--sentinel=preserved"
  ENVIRONMENT "SCANNER_SENTINEL=preserved"
)
set_target_properties(registration_test PROPERTIES LINKER_LANGUAGE CXX)

# Compare with the standard scanner's existing single-line behavior.
add_executable(reference_test reference.cpp)
set_target_properties(reference_test PROPERTIES LINKER_LANGUAGE CXX)
gtest_add_tests(
  TARGET reference_test
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reference.cpp"
  TEST_PREFIX "reference."
  EXTRA_ARGS "--sentinel=preserved"
  TEST_LIST _reference_tests
)
set_tests_properties(${_reference_tests}
  PROPERTIES ENVIRONMENT "SCANNER_SENTINEL=preserved")

# Existing callers apply these properties during configuration.
get_property(_registered_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_registered_tests}
  PROPERTIES TIMEOUT 17 RUN_SERIAL TRUE LABELS "scanner;fixture")

get_target_property(_compile_sources registration_test SOURCES)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compile-sources.txt" "${_compile_sources}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/executables.txt"
  CONTENT "$<TARGET_FILE:registration_test>\n$<TARGET_FILE:reference_test>\n")
"""


@unittest.skipUnless(_CMAKE and _CTEST and _NINJA, "CMake, CTest, and Ninja are required")
class CcTestRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="xllm cc_test ")
        self.addCleanup(self._temporary.cleanup)
        self._root = Path(self._temporary.name)
        self._source = self._root / "source"
        self._build = self._root / "build"
        (self._source / "a").mkdir(parents=True)
        (self._source / "b").mkdir()
        self._relative_source = self._source / "a" / "same_test.cpp"
        self._absolute_source = self._source / "b" / "same_test.cpp"

        multiline = "".join(f"{macro}(\n\t{suite},\n    {name}\n) {{}}\n" for macro, suite, name in _CASES)
        self._relative_source.write_bytes(multiline.replace("\n", "\r\n").encode())
        self._absolute_source.write_text(f"TEST(SecondSourceSuite,\n     SeparateCase) {{}}\n{_LITERALS}\n")
        reference = "".join(f"{macro}({suite}, {name}) {{}}\n" for macro, suite, name in _CASES)
        (self._source / "reference.cpp").write_text(reference + "TEST(SecondSourceSuite, SeparateCase) {}\n")
        (self._source / "CMakeLists.txt").write_text(_PROJECT)
        self._configure()

        # CTest only needs existing executable paths while listing. These files
        # are deliberately invalid executables and no test body is invoked.
        for path in (self._build / "executables.txt").read_text().splitlines():
            executable = Path(path)
            executable.write_text("This registration fixture must never execute.\n")
            executable.chmod(0o755)

    def _run(self, arguments: list[str]) -> str:
        result = subprocess.run(arguments, cwd=self._root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, f"{arguments}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def _configure(self) -> None:
        self._run(
            [
                _CMAKE,
                "-S",
                str(self._source),
                "-B",
                str(self._build),
                "-G",
                "Ninja",
                f"-DXLLM_CC_TEST_MODULE_DIR:PATH={_REPO_ROOT / 'cmake'}",
            ]
        )

    def _inventory(self) -> list[dict[str, Any]]:
        output = self._run([_CTEST, "--test-dir", str(self._build), "--show-only=json-v1"])
        return json.loads(output)["tests"]

    def _canonical(self, test: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        return {
            "name": test["name"].removeprefix(prefix),
            "command": test["command"][1:],
            "properties": sorted(test["properties"], key=lambda item: item["name"]),
        }

    def _regenerate(self) -> None:
        self._run([_CMAKE, "--build", str(self._build), "--target", "build.ninja"])

    def _configure_count(self) -> int:
        return int((self._build / "configure-count.txt").read_text())

    def test_multiline_cases_preserve_registration_and_properties(self) -> None:
        inventory = self._inventory()
        reference = {
            test["name"].removeprefix("reference."): test for test in inventory if test["name"].startswith("reference.")
        }
        actual = {test["name"]: test for test in inventory if not test["name"].startswith("reference.")}
        self.assertEqual(len(actual), 9)
        self.assertEqual(actual.keys(), reference.keys())
        for name, test in actual.items():
            with self.subTest(name=name):
                self.assertEqual(self._canonical(test), self._canonical(reference[name], "reference."))
                self.assertIn("--sentinel=preserved", test["command"])
                properties = {item["name"]: item["value"] for item in test["properties"]}
                self.assertEqual(properties["ENVIRONMENT"], ["SCANNER_SENTINEL=preserved"])
                self.assertEqual(properties["TIMEOUT"], 17)
                self.assertTrue(properties["RUN_SERIAL"])

        disabled = [
            test
            for test in actual.values()
            if any(item["name"] == "DISABLED" and item["value"] for item in test["properties"])
        ]
        self.assertEqual(len(disabled), 3)
        self.assertIn(
            "--gtest_filter=*/ParameterSuite.ParameterCase/*", actual["*/ParameterSuite.ParameterCase/*"]["command"]
        )
        self.assertIn("--gtest_filter=TypedSuite/*.TypedCase", actual["TypedSuite/*.TypedCase"]["command"])
        self.assertIn(
            "--gtest_filter=*/TypedParameterSuite/*.TypedParameterCase",
            actual["*/TypedParameterSuite/*.TypedParameterCase"]["command"],
        )

    def test_discovery_copies_preserve_literals_and_original_compile_sources(self) -> None:
        copies = list((self._build / "gtest-discovery").glob("*.cpp"))
        self.assertEqual(len(copies), 2)
        expected = {
            re.sub(b"[\r\n\t]", b" ", source.read_bytes().replace(b"\r\n", b"\n"))
            for source in (self._relative_source, self._absolute_source)
        }
        self.assertEqual({copy.read_bytes() for copy in copies}, expected)
        compile_sources = (self._build / "compile-sources.txt").read_text().split(";")
        self.assertEqual(compile_sources, ["a/same_test.cpp", str(self._absolute_source)])

        ninja = (self._build / "build.ninja").read_text().replace("$ ", " ")
        rerun = next(line for line in ninja.splitlines() if line.startswith("build build.ninja: RERUN_CMAKE"))
        self.assertIn(str(self._relative_source), rerun)
        self.assertIn(str(self._absolute_source), rerun)
        for copy in copies:
            self.assertNotIn(str(copy), rerun)

    def test_original_source_changes_regenerate_without_a_loop(self) -> None:
        copies = list((self._build / "gtest-discovery").glob("*.cpp"))
        initial_timestamps = {copy: copy.stat().st_mtime_ns for copy in copies}
        self._configure()
        self.assertEqual({copy: copy.stat().st_mtime_ns for copy in copies}, initial_timestamps)
        initial_count = self._configure_count()
        self._regenerate()
        self._regenerate()
        self.assertEqual(self._configure_count(), initial_count)

        with self._relative_source.open("ab") as source:
            source.write(b"\r\nTEST(AddedSuite,\r\n     AddedCase) {}\r\n")
        os.utime(self._relative_source, None)
        self._regenerate()
        self.assertEqual(self._configure_count(), initial_count + 1)
        self.assertIn("AddedSuite.AddedCase", {test["name"] for test in self._inventory()})
        changed_timestamps = {copy: copy.stat().st_mtime_ns for copy in copies}
        self._regenerate()
        self._regenerate()
        self.assertEqual(self._configure_count(), initial_count + 1)
        self.assertEqual({copy: copy.stat().st_mtime_ns for copy in copies}, changed_timestamps)
        self.assertEqual(list(self._build.rglob("*.o")), [])


if __name__ == "__main__":
    unittest.main()
