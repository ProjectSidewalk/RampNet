"""Compare two JSON artifacts exactly, or with a relative tolerance on floats only (#251).

Several checks prove that a committed JSON result is what the code produces today by
regenerating it and comparing bytes. Bytes are the strongest statement, and they hold on
the Linux CI runners, but a float written by ``json.dumps`` is the shortest repr of the
double, so a last-bit difference in a bootstrap sum on another platform's BLAS / libm
changes the text. On macOS that failed ``benchmark_power_135 --check`` with 110 floats off
by at most 6.66e-16, and the far-field forensics byte test likewise (#251 part 4).

:func:`compare_json` keeps the strongest statement available:

* **byte-identical** -- the bytes match; nothing else is consulted;
* **within tolerance** -- the bytes differ, but (a) the two texts are identical once every
  JSON number token is masked, so indentation, key order, line endings and every string
  match byte for byte, and (b) the parsed values match leaf by leaf: strings, booleans,
  null, ints and the types of all of them exactly, and floats within ``rel`` (relative, 1e-12
  by default) or ``abs_tol`` (absolute, 0 by default). NaN equals NaN;
* **differs** -- anything else, reported with the first differing key path, the number of
  differing leaves and the largest absolute and relative float difference.

Pass ``exact=True`` to require bytes (a CRLF copy of an LF file then fails, as it should
when the point of the test is the on-disk form).

Usage::

    from rampnet.json_compare import assert_json_equal_within
    cmp = assert_json_equal_within("regenerated.json", "committed.json")   # raises on differs
    print(cmp.summary())        # "byte-identical" or "within rel 1e-12: 110 float leaves ..."

Stdlib only.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union

DEFAULT_REL = 1e-12

#: A JSON number token (RFC 8259). Matched on the raw text to build the masked skeleton; a
#: digit run inside a string is masked too, which is harmless because strings are then
#: compared exactly as parsed leaves.
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

ABSENT = "<absent>"


def _is_float(x) -> bool:
    return isinstance(x, float)


def _same_exact(a, b) -> bool:
    if _is_float(a) and _is_float(b) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def diff_leaves(a, b, path: str = "$") -> Iterator[tuple]:
    """Yield ``(path, a_value, b_value)`` for every leaf that is not exactly equal.

    Dicts are walked in sorted key order and lists by index, so for a file written with
    ``sort_keys=True`` the first item is the first difference in the file. A key present on
    one side only, a length mismatch or a type mismatch (including ``42`` vs ``42.0``) is
    yielded once at that node, with the missing side as ``"<absent>"``.

    >>> list(diff_leaves({"a": [1, 2.0]}, {"a": [1, 2.5]}))
    [('$.a[1]', 2.0, 2.5)]
    """
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}"
            if k not in a or k not in b:
                yield p, a.get(k, ABSENT), b.get(k, ABSENT)
            else:
                yield from diff_leaves(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            p = f"{path}[{i}]"
            if i >= len(a) or i >= len(b):
                yield p, a[i] if i < len(a) else ABSENT, b[i] if i < len(b) else ABSENT
            else:
                yield from diff_leaves(a[i], b[i], p)
    elif not _same_exact(a, b):
        yield path, a, b


@dataclass
class Comparison:
    """The outcome of :func:`compare_json`. ``status`` is ``"identical"`` (bytes),
    ``"within"`` (float drift inside the tolerance only) or ``"differs"``."""
    status: str
    rel: float
    abs_tol: float
    n_bytes: tuple = (0, 0)
    first_byte_offset: int = -1
    formatting_differs: bool = False        # the number-masked skeletons differ
    n_float_diffs: int = 0                  # float leaves that differ at all
    n_beyond: int = 0                       # float leaves outside the tolerance
    n_structural: int = 0                   # every other differing leaf (type, str, int, key)
    max_abs: float = 0.0
    max_rel: float = 0.0
    first_key: str = ""                     # first leaf that differs at all
    first_beyond_key: str = ""              # first leaf that breaks the comparison
    first_beyond: tuple = field(default=())  # (a_value, b_value) at first_beyond_key

    @property
    def ok(self) -> bool:
        return self.status in ("identical", "within")

    def summary(self) -> str:
        if self.status == "identical":
            return "byte-identical"
        stats = (f"{self.n_float_diffs} float leaves differ, max |d| {self.max_abs!r}, "
                 f"max rel {self.max_rel!r}")
        if self.status == "within":
            return (f"within rel {self.rel:g}" + (f" / abs {self.abs_tol:g}" if self.abs_tol else "")
                    + f": {stats}; first at {self.first_key}")
        why = []
        if self.formatting_differs:
            why.append("formatting differs (not only number spellings)")
        if self.n_structural:
            why.append(f"{self.n_structural} non-float or structural leaf difference(s)")
        if self.n_beyond:
            why.append(f"{self.n_beyond} float leaf(s) beyond rel {self.rel:g}")
        where = (f"; first breaking key {self.first_beyond_key} "
                 f"({_short(self.first_beyond[0])} vs {_short(self.first_beyond[1])})"
                 if self.first_beyond_key else "")
        return "DIFFERS: " + "; ".join(why or ["bytes differ"]) + f"; {stats}{where}"


def _short(x, limit: int = 120) -> str:
    r = repr(x)
    return r if len(r) <= limit else r[:limit] + f"... ({len(r)} chars)"


def _bytes(x: Union[str, Path, bytes]) -> bytes:
    return x if isinstance(x, bytes) else Path(x).read_bytes()


def compare_json(a: Union[str, Path, bytes], b: Union[str, Path, bytes], *,
                 rel: float = DEFAULT_REL, abs_tol: float = 0.0,
                 exact: bool = False) -> Comparison:
    """Compare two JSON documents given as paths or raw bytes. See the module docstring.

    >>> compare_json(b'{"x": 0.1}\\n', b'{"x": 0.1}\\n').status
    'identical'
    >>> compare_json(b'{"x": 0.30000000000000004}\\n', b'{"x": 0.3}\\n').status
    'within'
    >>> compare_json(b'{"x": 0.3}\\r\\n', b'{"x": 0.3}\\n').status
    'differs'
    """
    ba, bb = _bytes(a), _bytes(b)
    c = Comparison("identical", rel, abs_tol, n_bytes=(len(ba), len(bb)))
    if ba == bb:
        return c
    c.first_byte_offset = next((i for i, (x, y) in enumerate(zip(ba, bb)) if x != y),
                               min(len(ba), len(bb)))
    c.formatting_differs = _NUMBER.sub(b"0", ba) != _NUMBER.sub(b"0", bb)
    for path, x, y in diff_leaves(json.loads(ba), json.loads(bb)):
        c.first_key = c.first_key or path
        if _is_float(x) and _is_float(y) and math.isfinite(x) and math.isfinite(y):
            d = abs(x - y)
            r = d / max(abs(x), abs(y))            # > 0, since x != y
            c.n_float_diffs += 1
            c.max_abs, c.max_rel = max(c.max_abs, d), max(c.max_rel, r)
            if not math.isclose(x, y, rel_tol=rel, abs_tol=abs_tol):
                c.n_beyond += 1
            else:
                continue
        else:
            c.n_structural += 1
        if not c.first_beyond_key:
            c.first_beyond_key, c.first_beyond = path, (x, y)
    broken = exact or c.formatting_differs or c.n_structural or c.n_beyond
    c.status = "differs" if broken else "within"
    if exact and not c.first_beyond_key:
        c.first_beyond_key = c.first_key or "<bytes>"
        c.first_beyond = ("<bytes>", "<bytes>")
    return c


def assert_json_equal_within(path_a, path_b, rel: float = DEFAULT_REL, *,
                             abs_tol: float = 0.0, exact: bool = False) -> Comparison:
    """:func:`compare_json`, raising ``AssertionError`` with the summary unless it passes.

    Returns the :class:`Comparison`, so a caller can still say whether the match was
    byte-identical (``cmp.status == "identical"``) or only within tolerance.
    """
    c = compare_json(path_a, path_b, rel=rel, abs_tol=abs_tol, exact=exact)
    if not c.ok:
        raise AssertionError(f"{path_a if not isinstance(path_a, bytes) else '<bytes>'} vs "
                             f"{path_b if not isinstance(path_b, bytes) else '<bytes>'}: "
                             + c.summary())
    return c
