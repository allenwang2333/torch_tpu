#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Measures error message test coverage for torch_tpu.

USAGE:
  third_party/py/torch_tpu/google/tools/error_coverage.py > report.txt

    Collects and prints the test coverage for torch_tpu error messages, file by
    file. If it's infeasible to test an error macro (e.g. because the function
    never returns an error Status), please add an
    `// ERROR_COV_INFEASIBLE=<why>` comment on the same line as the error macro
    so that it will be excluded from the coverage report.

  third_party/py/torch_tpu/google/tools/error_coverage.py analyze < report.txt

    Analyzes the coverage report and prints the summary and a CSV for filing
    bugs.
"""

import bisect
import collections
from collections.abc import Mapping
import concurrent.futures
import os
import re
import subprocess
import sys


# All error message coverage targets. Shared with run_errors_test.py, which runs
# the same targets as data dependencies inside a blaze test.
# TODO(wan): Add *_cov versions of
# //third_party/py/torch_tpu/tests/distributed:device_mesh_errors_test and
# //third_party/py/torch_tpu/tests/distributed:handshake_errors_test, and run
# them in error_coverage_test.
COV_TEST_TARGETS = (
    # go/keep-sorted start
    "//third_party/py/torch_tpu/tests/distributed:collective_errors_test_cov",
    "//third_party/py/torch_tpu/tests:errors_test_cov",
    "//third_party/py/torch_tpu/tests:tpu_errors_test_cov",
    # go/keep-sorted end
)

# These macros create a new error from scratch (as opposed to
# transforming/plumbing an existing error). We focus on coverage for such macros
# for now.
_NEED_TO_COVER_MACROS = (
    "TT_ERROR",
    "TT_RET_CHECK",
    "TT_CHECK_THROW",
)

def cd_google3() -> None:
  """Changes to the google3 directory."""
  while os.path.basename(os.getcwd()) != "google3":
    if os.getcwd() == "/":
      sys.exit(
          "The current directory is not under google3. Please run this script"
          " from within a google3 directory."
      )
    os.chdir("..")


def get_torch_tpu_cpp_files() -> list[str]:
  """Returns the list of torch_tpu C++ files."""
  torch_tpu_dir = "third_party/py/torch_tpu"
  torch_tpu_cpp_files = []
  for root, dirs, files in os.walk(torch_tpu_dir):
    # Prune hidden directories and build output symlinks (e.g. blaze-*, bazel-*)
    # in-place so os.walk does not recurse into them.
    dirs[:] = [
        d
        for d in dirs
        if not d.startswith(".")
        and not d.startswith("blaze-")
        and not d.startswith("bazel-")
    ]
    for file in files:
      if file.endswith(".cc") or file.endswith(".h"):
        torch_tpu_cpp_files.append(os.path.join(root, file))
  return torch_tpu_cpp_files


class ErrorMacroLine:
  """Represents a line containing an error macro."""

  def __init__(self, line_num: int, cov_feasible: bool, macro: str):
    """Creates an ErrorMacroLine object.

    Args:
      line_num: The 1-based line number of the error macro.
      cov_feasible: Whether the error macro is feasible to test.
      macro: The name of the error macro.
    """

    self.line_num = line_num
    self.cov_feasible = cov_feasible
    self.macro = macro


def get_file_to_locs() -> dict[str, list[ErrorMacroLine]]:
  """Scans all torch_tpu C++ files for error macros."""
  cd_google3()
  file_to_locs: dict[str, list[ErrorMacroLine]] = {}
  cpp_files = get_torch_tpu_cpp_files()
  with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(lambda f: (f, get_error_macro_locs(f)), cpp_files)
    for cpp_file, locs in results:
      if locs:
        file_to_locs[cpp_file] = locs
  return file_to_locs


class SourceLoc:
  """Represents a source location in a C++ file."""

  def __init__(self, path: str, line_number: int):
    self.path = path
    self.line_number = line_number

  def __str__(self):
    return f"{self.path}:{self.line_number}"

  def __repr__(self):
    return self.__str__()

  def __lt__(self, other):
    """Compares two SourceLoc objects."""
    if not isinstance(other, SourceLoc):
      raise TypeError("Can only compare SourceLoc objects.")
    return self.path < other.path or (
        self.path == other.path and self.line_number < other.line_number
    )

  def __eq__(self, other):
    """Compares two SourceLoc objects."""
    if not isinstance(other, SourceLoc):
      raise TypeError("Can only compare SourceLoc objects.")
    return self.path == other.path and self.line_number == other.line_number

  def __hash__(self):
    return hash((self.path, self.line_number))


# Matches one of the torch_tpu error macros.
_TT_ERROR_API_RE = re.compile(
    r"\b(TT_ERROR|TT_RET_CHECK|TT_ASSIGN_OR_RETURN|TT_ASSIGN_OR_THROW|"
    r"TT_RETURN_IF_ERROR|TT_THROW_IF_ERROR|TT_CHECK_THROW)\("
)


# Matches an ERROR_COV_INFEASIBLE comment.
_COV_INFEASIBLE_RE = re.compile(r".*\bERROR_COV_INFEASIBLE\s*=.+")


def get_error_macro_locs(cpp_file: str) -> list[ErrorMacroLine]:
  """Scans a C++ file for error macros.

  Args:
    cpp_file: The C++ filepath to scan.

  Returns:
    The 1-based line numbers of the error macros found in the file, or an empty
    list if the file is a test file. The numbers are 1-based and sorted in
    ascending order.
  """

  if cpp_file.endswith(("_test.cc", "_test_helper.h")):
    return []
  macro_lines = []
  in_preprocessor = False
  with open(cpp_file, "r") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()

      # Ignore comments and preprocessor directives.
      if in_preprocessor or line.startswith("#"):
        in_preprocessor = line.endswith("\\")
        continue
      if line.startswith(("//", '"')):
        continue

      m = _TT_ERROR_API_RE.search(line)
      if m:
        # If we cannot find the ERROR_COV_INFEASIBLE comment, we assume the
        # error macro is feasible to test.
        cov_feasible = _COV_INFEASIBLE_RE.match(line) is None
        macro_lines.append(ErrorMacroLine(line_num, cov_feasible, m.group(1)))
  return macro_lines


# Matches the covered locations output by the errors_test_cov target.
# Group 1: file path.
# Group 2: line number.
_COVERED_LOCS_RE = re.compile(r"Covered ([/\w\.]+):(\d+)")


def parse_cov_output(cov_output: str) -> set[SourceLoc]:
  """Parses the coverage output from the coverage target.

  Args:
    cov_output: The stdout of the coverage target.

  Returns:
    The set of covered source locations.
  """
  return {
      SourceLoc(path, int(line_num))
      for path, line_num in _COVERED_LOCS_RE.findall(cov_output)
  }


class CoverageForFile:
  """Represents the coverage for a single C++ file."""

  def __init__(
      self,
      cpp_file: str,
      locs: list[int],
      covered_locs: list[int],
      infeasible_locs: list[int],
  ):
    """Initializes the coverage for a single C++ file.

    Args:
      cpp_file: The C++ filepath.
      locs: The 1-based line numbers of the error macros found in the file.
      covered_locs: The 1-based line numbers of the error macros covered by the
        error tests.
      infeasible_locs: The 1-based line numbers of the error macros that are
        infeasible to test.
    """

    self.cpp_file = cpp_file
    self.locs = sorted(locs)
    self.covered_locs = sorted(covered_locs)
    self.infeasible_locs = sorted(infeasible_locs)
    # Locations of error macros that are not covered by the error tests.
    self.not_covered_locs = sorted(set(locs) - set(covered_locs))
    # Locations of error macros that we should add tests for.
    self.to_cover_locs = sorted(
        set(self.not_covered_locs) - set(infeasible_locs)
    )
    # Locations of error macros that are marked as infeasible to test but
    # actually tested.
    self.infeasible_and_covered_locs = sorted(
        set(self.infeasible_locs) & set(self.covered_locs)
    )

  def ratio(self) -> float:
    """Returns the coverage ratio as a float in the range [0, 1]."""
    return len(self.covered_locs) / len(self.locs) if self.locs else 1.0

  def ratio_upper_bound(self) -> float:
    """Returns the maximum possible coverage ratio.

    Since some error macros are infeasible to test, maximum possible
    coverage may be < 1.
    """
    return (
        len(set(self.locs) - set(self.infeasible_locs)) / len(self.locs)
        if self.locs
        else 1.0
    )

  def __str__(self):
    msg = (
        f"{self.cpp_file}:"
        f" {len(self.covered_locs)}/{len(self.locs)} ({100*self.ratio():.1f}%)"
        " error macros covered"
    )
    if self.infeasible_locs:
      msg += f" ({len(self.infeasible_locs)} error macros infeasible to cover)"
    return msg

  def __repr__(self):
    return self.__str__()

  def __lt__(self, other):
    """Compares two CoverageForFile objects.

    Sorts by the number of to-be-covered locations in ascending order.
    Breaks ties by the number of covered locations in descending order.

    With this ordering, smaller CoverageForFile objects in general need less
    work.

    Args:
      other: The other CoverageForFile object to compare to.

    Returns:
      True if self is less than other, False otherwise.
    """
    if not isinstance(other, CoverageForFile):
      raise TypeError("Can only compare CoverageForFile objects.")

    return (len(self.to_cover_locs), -len(self.covered_locs)) < (
        len(other.to_cover_locs),
        -len(other.covered_locs),
    )

  def to_dict(self) -> Mapping[str, int]:
    """Returns the serialized statistics map for the file."""
    return {
        "infeasible": len(self.infeasible_locs),
        "remaining": len(self.to_cover_locs),
        "total": len(self.locs),
    }


def sub_coverage_report(file_path: str, lines: list[int], status: str) -> str:
  """Generates a human-readable report for lines with a given status.

  Args:
    file_path: The C++ file path.
    lines: A list of 1-based line numbers to report.
    status: The status of the lines (e.g. "covered").

  Returns:
    A human-readable report for the given lines.
  """
  report = []
  for loc in lines:
    report.append(f"  {file_path}:{loc}: {status}")
  return "\n".join(report)


def coverage_report(coverage: CoverageForFile) -> str:
  """Generates a human-readable report for a single file's error coverage."""
  report = [
      str(coverage),
      sub_coverage_report(coverage.cpp_file, coverage.covered_locs, "covered"),
      sub_coverage_report(
          coverage.cpp_file, coverage.infeasible_locs, "infeasible to cover"
      ),
      sub_coverage_report(
          coverage.cpp_file, coverage.to_cover_locs, "TO BE COVERED"
      ),
  ]
  return "\n".join(report)


def _need_to_cover(loc: ErrorMacroLine) -> bool:
  """Returns whether the loc.macro needs to be covered."""
  return loc.macro in _NEED_TO_COVER_MACROS


def get_per_file_coverage(
    file_to_locs: dict[str, list[ErrorMacroLine]],
    covered_raw_locs: set[SourceLoc],
) -> list[CoverageForFile]:
  """Returns the covered error macro locations for each file."""
  file_to_covered_raw_locs: dict[str, set[int]] = collections.defaultdict(set)
  for raw_loc in covered_raw_locs:
    file_to_covered_raw_locs[raw_loc.path].add(raw_loc.line_number)

  result = []
  for cpp_file, locs in file_to_locs.items():
    covered_locs = []
    covered_raw_lines = file_to_covered_raw_locs.get(cpp_file)
    if covered_raw_lines:
      loc_line_nums = [loc.line_num for loc in locs]
      for line_num in covered_raw_lines:
        # Find the largest line number in locs that is smaller than or equal to
        # the covered line number. This is necessary as the line numbers in locs
        # are for the lines containing the *start* of the error macros, while the
        # line numbers in covered_raw_locs are for the lines containing the
        # *end* (i.e. the closing parenthesis) of the error macros.
        idx = bisect.bisect_right(loc_line_nums, line_num) - 1
        if idx >= 0:
          loc = locs[idx]
          if _need_to_cover(loc):
            covered_locs.append(loc.line_num)

    need_to_cover_locs = [l.line_num for l in locs if _need_to_cover(l)]
    if not need_to_cover_locs:
      continue

    result.append(
        CoverageForFile(
            cpp_file,
            locs=need_to_cover_locs,
            covered_locs=covered_locs,
            infeasible_locs=[
                l.line_num
                for l in locs
                if not l.cov_feasible and _need_to_cover(l)
            ],
        )
    )
  return result


def print_coverage_report() -> None:
  """Collects the raw coverage data and prints them to stdout."""

  print("Scanning torch_tpu C++ files...", file=sys.stderr, flush=True)
  file_to_locs = get_file_to_locs()
  print(
      f"Scanned {len(file_to_locs)} torch_tpu C++ files.",
      file=sys.stderr,
      flush=True,
  )

  covered_locs = set()

  # Run the tests in coverage mode, capturing the stdout.
  print(
      "Error tests running. This may take ~5 minutes...",
      file=sys.stderr,
      flush=True,
  )
  cov_run = subprocess.run(
      [
          "blaze",
          "test",
          "--cache_test_results=no",
          "--test_output=all",
          "--test_env=TORCH_TPU_ENABLE_DEBUG_CHECKS=1",
      ]
      + list(COV_TEST_TARGETS),
      capture_output=True,
      check=False,
  )
  if cov_run.returncode != 0:
    sys.exit(
        "ERROR: error tests failed. Please check the test log to make sure"
        f" it's running correctly:\n{cov_run.stderr.decode('utf-8')}"
    )
  cov_stdout = cov_run.stdout.decode("utf-8")
  locs = parse_cov_output(cov_stdout)
  print(
      f"Error tests cover {len(locs)} error macro locations.",
      file=sys.stderr,
      flush=True,
  )
  if not locs:
    sys.exit(
        "ERROR: Error tests did not cover any error macro locations. "
        "Please check the test log to make sure it's running correctly:\n"
        f"{cov_stdout}",
    )
  covered_locs.update(locs)

  # Compute and print the coverage for each file, sorted by the number of
  # not-covered locations in ascending order. I.e. the files that need most
  # work are listed last.
  print(
      f"All error tests cover {len(covered_locs)} error macro locations:",
      flush=True,
  )
  per_file_coverage = sorted(get_per_file_coverage(file_to_locs, covered_locs))
  for coverage in per_file_coverage:
    print(coverage_report(coverage), flush=True)

  # If any line marked as ERROR_COV_INFEASIBLE is actually covered, print a
  # warning.
  for coverage in per_file_coverage:
    for loc in coverage.infeasible_and_covered_locs:
      print(
          f"{coverage.cpp_file}:{loc}: WARNING: the line is marked as"
          " infeasible to test but actually tested. Please remove"
          " the ERROR_COV_INFEASIBLE comment.",
          file=sys.stderr,
          flush=True,
      )


# Matches a line like
# third_party/py/torch_tpu/ops/addcmul/addcmul_aten_kernels.cc: 1/8 (12.5%)
# error macros covered (1 error macros infeasible to cover)
#
# Capture groups:
# 1. file path
# 2. number of covered macros
# 3. number of all macros
# 5. number of infeasible macros
_FILE_SUMMARY_RE = re.compile(
    r"([^\s]+):\s*(\d+)/(\d+).*macros covered(.*?(\d+) error macros"
    r" infeasible.*)?"
)


def analyze_coverage_report() -> None:
  """Analyzes the coverage report provided on stdin and prints the summary and a CSV for filing bugs."""

  # For each C++ source file that is not fully covered (covered locs +
  # infeasible locs < total locs), file a bug:
  # Title = [error message] path/to/file
  # Description = Add tests to cover N TT_* error macros in path/to/file.
  #               See go/tt-fixit-errors for details.
  # Assignee = <empty>
  # Component ID = 1848883
  # FixIt Points = uncovered locs / 10 (round up)

  print("Bug Title,Bug Description,Assignee,Component ID,FixIt Points")
  for line in sys.stdin:
    m = _FILE_SUMMARY_RE.match(line)
    if not m:
      continue
    file_path = m.group(1)
    covered_macros = int(m.group(2))
    total_macros = int(m.group(3))
    infeasible_macros = int(m.group(5)) if m.group(5) else 0
    macros_to_cover = total_macros - covered_macros - infeasible_macros
    if macros_to_cover <= 0:
      continue

    fixit_points = (macros_to_cover - 1) // 10 + 1
    print(
        f"[error message] {file_path},Add tests to cover {macros_to_cover} TT_*"
        f" error macros in {file_path}. See go/tt-fixit-errors for"
        f" details.,,1848883,{fixit_points}"
    )


def main():
  args = sys.argv[1:]
  if not args:
    print_coverage_report()
  elif len(args) == 1 and args[0] == "analyze":
    analyze_coverage_report()
  else:
    sys.exit(
        "ERROR: Invalid command line arguments. See the usage below:\n"
        f"{__doc__}"
    )


if __name__ == "__main__":
  main()
