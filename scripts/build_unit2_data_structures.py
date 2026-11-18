"""Builder for unit2_data_structures.ipynb (Unit II of the syllabus)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nbhelper import md, code, write_notebook

cells = []

# ---------------------------------------------------------------------------
cells.append(md(r"""
# Unit II — Fundamentals of Python Programming 2: Data Structures, Functional Tools & Modules

Python's four built-in container types — **list, tuple, set, dictionary** — are the vocabulary of every data-science program. This notebook is a long-lived reference: it documents how each structure is created, indexed, mutated and iterated, then moves on to the functional tools (`map`, `filter`, `reduce`, `zip`, `enumerate`) that process entire iterables at once, and finally to the **module system** that lets you organise and reuse your own code. Every example is small and runnable offline; data-science flavoured tasks (word counts, grouping records) appear near the end.

## Learning objectives
- Distinguish **sequences** from **collections**, and mutable from immutable / hashable types.
- Create lists and use indexing, negative indices and slicing with a step; mutate lists with `append`, `extend`, `insert`, `remove`, `pop`, `clear`, `del`, `index`, `count`, `sort`, `sorted`, `reverse`, `copy`.
- Write list comprehensions with conditions and nested loops, and explain shallow vs deep copying and aliasing pitfalls.
- Create tuples, explain immutability, unpack values, build single-element tuples, use tuple methods and decide when a tuple beats a list.
- Create sets, use `add`/`remove`/`discard`/`pop`, perform union/intersection/difference/symmetric difference, test subsets, use `frozenset`, and deduplicate data.
- Create dictionaries, use `keys`/`values`/`items`/`get`/`setdefault`/`update`/`pop`/`popitem`, write dict comprehensions, iterate and merge dicts, build nested dicts, and recall ordering guarantees.
- Use `defaultdict` and `Counter` from `collections`.
- Apply `map()`, `filter()`, `reduce()` (from `functools`), `zip()` / `zip(*)`, `enumerate()`, `all()` / `any()`, and `sorted(..., key=...)` to iterables.
- Compare comprehensions with `map`/`filter` and choose between them.
- Explain what a module is, why modules matter, every common `import` form, aliases, `__name__ == "__main__"`, and the module search path.
- Tour commonly used standard-library modules: `math`, `random`, `statistics`, `datetime`, `collections`, `itertools`, `functools`, `os`, `sys`, `json`, `csv`, `re`, `pathlib`.
- Create a user-defined module from the notebook, import it, and explain packages and `__init__.py`.
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Official documentation & references

All links below point to authoritative official sources.

- [Python 3 tutorial — Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Python 3 — Built-in Types (list, tuple, set, dict)](https://docs.python.org/3/library/stdtypes.html)
- [Python 3 — `functools` (reduce, lru_cache, partial)](https://docs.python.org/3/library/functools.html)
- [Python 3 tutorial — Modules](https://docs.python.org/3/tutorial/modules.html)
- [Python 3 — `collections` (Counter, defaultdict, deque)](https://docs.python.org/3/library/collections.html)
- [Python 3 — `itertools`](https://docs.python.org/3/library/itertools.html)
- [Python 3 — `pathlib`](https://docs.python.org/3/library/pathlib.html)
- [Python 3 — `re` regular expressions](https://docs.python.org/3/library/re.html)
- [Python 3 — `csv` module](https://docs.python.org/3/library/csv.html)
- [Python 3 — `json` module](https://docs.python.org/3/library/json.html)
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from pathlib import Path
DATA = Path("../../data")          # local CSVs live two levels above this notebook

# Standard-library imports used throughout this notebook.
import math
import random
import statistics
import datetime as dt
import collections
import itertools
import functools
import os
import sys
import json
import csv
import io
import re
import importlib

print("Python data-structures notebook ready.")
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 1. Sequences vs collections

Python groups containers in two broad families:

- **Sequences** are ordered and indexed by integer position: `str`, `list`, `tuple`, `range`, `bytes`. They support `len()`, indexing `x[i]`, slicing `x[i:j:k]`, `in`, concatenation `+` and repetition `*`.
- **Collections / unordered containers** are not indexable by position: `set` and `dict` (dict is ordered by *insertion* since Python 3.7, but has no integer indexing).

Two more keywords matter everywhere:

- **Mutable** objects can change after creation: `list`, `set`, `dict`.
- **Immutable** objects cannot: `str`, `tuple`, `int`, `float`, `frozenset`. Immutable *hashable* objects can be set members or dict keys.

`isinstance` is the runtime way to check a type; `type()` reports the exact class.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
samples = {"str": "abc", "list": [1, 2], "tuple": (1, 2),
           "set": {1, 2}, "dict": {"a": 1}, "range": range(3)}

for name, value in samples.items():
    print(f"{name:6} type={type(value).__name__:6}",
          f"len={len(value)}",
          f"sequence={isinstance(value, collections.abc.Sequence)}",
          f"iterable={isinstance(value, collections.abc.Iterable)}")

# Only immutable/hashable objects can be dictionary keys or set members.
print("hashable list?", isinstance([1], collections.abc.Hashable))
print("hashable tuple?", isinstance((1, 2), collections.abc.Hashable))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 2. Lists — creation, indexing and slicing

A **list** is an ordered, mutable sequence written with square brackets. Common constructors:

- `[]` — empty list
- `list(iterable)` — build from any iterable (string, range, set, dict → keys)
- `[expr for x in iterable]` — list comprehension
- `list(range(n))` — a list of integers

### Indexing
`L[i]` returns the item at position `i` (0-based). Negative indices count from the end: `L[-1]` is the last item, `L[-2]` the second-last. Out-of-range indices raise `IndexError`.

### Slicing `L[start:stop:step]`
Returns a **new list**. `start` is inclusive, `stop` is exclusive, `step` defaults to 1. A negative step walks backwards. Omitting a bound means "from the beginning"/"to the end". Note `L[:]` is a shallow copy of the whole list.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# Different ways to create lists.
empty = []
from_range = list(range(5))
from_string = list("hello")
from_dict_keys = list({"a": 1, "b": 2})   # iterating a dict yields keys
print(empty, from_range, from_string, from_dict_keys)

L = ["a", "b", "c", "d", "e"]
print("L[0] =", L[0], " L[-1] =", L[-1], " L[-2] =", L[-2])

# Slicing: start:stop:step (stop is exclusive).
print("L[1:4]   =", L[1:4])     # b c d
print("L[:3]    =", L[:3])      # a b c
print("L[3:]    =", L[3:])      # d e
print("L[::2]   =", L[::2])     # every second item
print("L[::-1]  =", L[::-1])    # reversed copy
print("L[4:0:-1]=", L[4:0:-1])  # backwards, stop exclusive
print("L[:]     =", L[:])       # shallow copy
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 3. Mutating a list

Lists are mutable. The in-place methods (they return `None`):

| Operation | Meaning |
|---|---|
| `L.append(x)` | add one item `x` to the end |
| `L.extend(it)` | append every item of iterable `it` |
| `L.insert(i, x)` | insert `x` *before* index `i` |
| `L.remove(x)` | remove the **first** value equal to `x` (raises `ValueError` if absent) |
| `L.pop([i])` | remove and return item at `i` (default last) |
| `L.clear()` | remove all items |
| `del L[i]` / `del L[i:j]` | delete by index or slice |
| `L[i] = x` / `L[i:j] = it` | replace by index or slice |

In-place methods mutate the object; they do **not** return a new list. `sorted(L)` and `L.copy()` do return new objects. Keeping this distinction straight prevents a very common bug (`x = L.sort()` gives `None`).
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
L = [1, 2]
L.append(3)                 # one element
L.extend([4, 5])            # many elements
L.insert(0, 99)             # insert before index 0
print("after add/extend/insert:", L)

L.remove(99)                # remove first matching value
last = L.pop()              # remove and return last
first = L.pop(0)            # remove and return at index 0
del L[0]                    # delete by index
print("removed:", first, last, " now:", L)

L.clear()
print("after clear:", L)

# In-place methods return None.
values = [3, 1, 2]
result = values.sort()
print("sort() returns:", result, " but values is now:", values)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 4. Searching, counting, sorting and reversing

- `L.index(x[, start[, end]])` — first index whose value equals `x` (`ValueError` if missing).
- `L.count(x)` — number of occurrences of `x`.
- `x in L` — membership test (returns `bool`).
- `L.sort(*, key=None, reverse=False)` — **in-place** stable sort.
- `sorted(iterable, *, key=None, reverse=False)` — returns a **new sorted list**.
- `L.reverse()` — reverses in place; `reversed(L)` returns an iterator.

`key` is a function applied to each element before comparison. Because sorting is *stable*, equal keys keep their original relative order — this enables multi-pass sorts (sort by secondary key first, primary key last).
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
nums = [4, 2, 7, 2, 9, 2]
print("index(7) =", nums.index(7))
print("count(2) =", nums.count(2))
print("5 in nums?", 5 in nums)

# sort in place vs sorted new list
a = [3, 1, 2]
b = sorted(a)
print("a after sorted(a):", a, "| b:", b, "| same object?", a is b)

words = ["Banana", "apple", "Cherry"]
print("case-insensitive:", sorted(words, key=str.lower))
print("reverse:", sorted([3, 1, 2], reverse=True))

# Sorting records by a field with a key function.
records = [("ana", 30), ("bob", 25), ("cara", 30)]
print("by age:", sorted(records, key=lambda r: r[1]))

# Stable multi-pass sort: by age, then by name as primary.
print("by age then name:",
      sorted(sorted(records, key=lambda r: r[0]), key=lambda r: r[1]))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 5. Copying, aliasing and memory pitfalls

Assignment (`b = a`) never copies data — it binds a second name to the **same object**. Mutating through one name is visible through the other.

- `L.copy()` or `L[:]` or `list(L)` → **shallow copy**: a new outer list, but nested objects are shared.
- `copy.deepcopy(L)` → **deep copy**: nested objects are copied recursively.

For a flat list of numbers/strings a shallow copy is enough. For nested lists/dicts you usually need `deepcopy` to avoid surprises.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
import copy

a = [1, 2, 3]
b = a              # alias: same object
b.append(4)
print("a =", a, "| a is b?", a is b)

shallow = a.copy()
shallow[0] = 99
print("a =", a, "| shallow =", shallow, "| outer shared?", a is shallow)

nested = [[1, 2], [3, 4]]
sh2 = nested.copy()
sh2[0].append(999)          # inner list is shared!
print("nested =", nested, "| sh2 =", sh2, "| inner shared?", nested[0] is sh2[0])

deep = copy.deepcopy(nested)
deep[0].append(777)
print("nested =", nested, "| deep =", deep, "| inner shared?", nested[0] is deep[0])
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 6. List comprehensions

Syntax: `[expression for item in iterable if condition]`.

They are more compact and usually faster than an explicit `for` + `append` loop. Optional pieces:

- A single `if` filter: `[x for x in xs if x > 0]`
- `if/else` in the **expression**: `[x if x > 0 else 0 for x in xs]`
- Multiple `for` clauses → nested iteration, evaluated left to right.
- Multiple `if` clauses after the `for`.

Nested loops in a comprehension produce a flat list; to get a matrix use a nested comprehension `[[... for j ...] for i ...]`.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
xs = range(10)
print("evens:", [x for x in xs if x % 2 == 0])
print("clamp negatives:", [x if x > 0 else 0 for x in [-2, -1, 0, 1, 2]])
print("two conditions:", [x for x in xs if x % 2 == 0 if x > 4])

# Nested iteration: cartesian product pairs.
pairs = [(i, j) for i in range(3) for j in range(2)]
print("pairs:", pairs)

# Nested comprehension -> matrix (list of lists).
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print("matrix:", matrix)

# Equivalent explicit loop, for comparison.
out = []
for x in xs:
    if x % 2 == 0:
        out.append(x)
print("loop result:", out)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 7. Tuples

A **tuple** is an ordered, **immutable** sequence written with parentheses. Once created you cannot add, remove or change elements.

- `t = (1, 2, 3)`, `t = 1, 2, 3` (packing), `tuple(iterable)`.
- **Single-element tuple needs a trailing comma**: `(5,)` — without the comma `(5)` is just the integer 5. Empty tuple: `()`.
- Methods: `t.count(x)` and `t.index(x)` — the only two tuple methods.
- Immutability makes tuples **hashable** (if all elements are hashable), so they can be dict keys or set members. Lists cannot.

**When to use a tuple:** fixed collections (coordinates, RGB colours, DB rows), dict keys, multiple return values from a function. Use a list when the collection must grow or change. The comma — not the parentheses — is what creates a tuple.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
t = (10, 20, 30)
packed = 1, 2, 3
single = (5,)          # trailing comma is required!
not_a_tuple = (5)
empty = ()
from_list = tuple([1, 2, 3])

print(t, packed, single, type(single).__name__, type(not_a_tuple).__name__, empty, from_list)
print("methods:", t.count(20), t.index(30))

# Tuples are immutable but can *contain* mutable objects.
t2 = ([1, 2], "x")
t2[0].append(3)
print("tuple with inner list mutated:", t2)

# Hashable -> usable as dict keys / set members.
lookup = {("Paris", "FR"): 2.1, ("Rome", "IT"): 2.8}
print("tuple-key dict:", lookup[("Rome", "IT")])
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 8. Tuple unpacking

Unpacking assigns each element to a separate variable: `a, b = (1, 2)`. The number of variables must match the length unless you use a starred target.

- Extended unpacking: `first, *rest = seq` — `rest` is a list.
- Ignore values with `_`: `_, second = pair`.
- Swapping values: `a, b = b, a` works without a temporary variable.
- `zip()` plus unpacking is the idiomatic way to walk parallel sequences.
- `divmod(17, 5)` returns a tuple `(quotient, remainder)` — a classic unpacking use.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
a, b, c = (1, 2, 3)
print("unpacked:", a, b, c)

first, *rest = [1, 2, 3, 4]
print("first =", first, "rest =", rest)

head, *middle, tail = "abcdefg"
print("head =", head, "middle =", ''.join(middle), "tail =", tail)

x, y = 5, 9
x, y = y, x
print("swapped:", x, y)

quotient, remainder = divmod(17, 5)
print("divmod ->", quotient, remainder)

name, _ = ("status", "ok")
print("name =", name)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 9. Sets

A **set** is an unordered collection of **unique, hashable** values written with braces. Sets are mutable but their elements must be immutable/hashable.

- Creation: `{1, 2, 3}`, `set(iterable)`, `set()` for empty (note `{}` is an empty **dict**).
- `s.add(x)` — add one element; `s.update(it)` — add many.
- `s.remove(x)` — remove, raising `KeyError` if absent; `s.discard(x)` — remove silently if absent.
- `s.pop()` — remove and return an arbitrary element; `s.clear()` — empty the set.
- Deduplication: `list(set(items))` removes duplicates but **does not preserve original order**; use `dict.fromkeys(items)` to dedupe while preserving order.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
s = {3, 1, 2, 3, 1}          # duplicates dropped
print("set:", s, "len:", len(s))
print("empty set:", set(), type({}).__name__)

s.add(4)
s.update([5, 6])
print("after add/update:", s)

s.discard(100)               # no error even though absent
print("after discard(100):", s)
removed = s.pop()            # arbitrary element
print("popped:", removed, "now:", s)

dupes = ["a", "b", "a", "c", "b"]
print("dedup unordered:", list(set(dupes)))
print("dedup ordered:", list(dict.fromkeys(dupes)))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 10. Set algebra, subsets and `frozenset`

Let `a = {1, 2, 3}` and `b = {3, 4, 5}`:

| Operation | Operator | Method | Result |
|---|---|---|---|
| Union | `a \| b` | `a.union(b)` | `{1,2,3,4,5}` |
| Intersection | `a & b` | `a.intersection(b)` | `{3}` |
| Difference | `a - b` | `a.difference(b)` | `{1,2}` |
| Symmetric difference | `a ^ b` | `a.symmetric_difference(b)` | `{1,2,4,5}` |
| Subset | `a <= b` | `a.issubset(b)` | `bool` |
| Superset | `a >= b` | `a.issuperset(b)` | `bool` |
| Disjoint | — | `a.isdisjoint(b)` | `bool` |

`frozenset` is an immutable set: it supports the set operations but no `add`/`remove`, and it is **hashable** so it can be a dict key or a member of another set.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
a, b = {1, 2, 3}, {3, 4, 5}
print("union:", a | b, a.union(b))
print("intersection:", a & b, a.intersection(b))
print("difference:", a - b, a.difference(b))
print("symmetric diff:", a ^ b, a.symmetric_difference(b))

print("is subset?", {1, 2} <= a, {1, 2}.issubset(a))
print("is superset?", a >= {1, 2})
print("disjoint?", a.isdisjoint({7, 8}))

f = frozenset([1, 2, 3])
print("frozenset:", f, "hashable:", isinstance(f, collections.abc.Hashable))
print("as dict key:", {f: "value"})
print("as set member:", {f, frozenset([4, 5])})
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 11. Dictionaries — creation and access

A **dict** maps hashable **keys** to arbitrary **values**. Creation:

- `{}` empty dict; `{"a": 1, "b": 2}` literal.
- `dict(a=1, b=2)` from keyword arguments.
- `dict([("a", 1), ("b", 2)])` from pairs.
- `{k: v for ...}` dict comprehension.

Access and views:

- `d[key]` — returns value, raises `KeyError` if missing.
- `d.get(key[, default])` — returns `default` (default `None`) if missing, never raises.
- `d.keys()`, `d.values()`, `d.items()` — dynamic **view objects** reflecting later changes.
- `key in d` — tests keys (fast, average O(1)).

Since Python 3.7 dicts **preserve insertion order** — a language guarantee, not an implementation detail.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
d = {"name": "Ada", "age": 36}
d2 = dict(name="Ada", age=36)
d3 = dict([("name", "Ada"), ("age", 36)])
print(d, d2, d3)
print("by key:", d["name"])
print("get missing:", d.get("city", "unknown"))

keys_view = d.keys()
values_view = d.values()
items_view = d.items()
print("keys:", list(keys_view), "values:", list(values_view), "items:", list(items_view))

d["city"] = "London"          # views update live
print("keys view after insert:", list(keys_view))
print("'age' in d?", "age" in d, "| 'London' in d?", "London" in d)
print("order preserved:", list(d))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 12. Dictionary methods

| Method | Behaviour |
|---|---|
| `d.setdefault(k, default)` | return `d[k]`; if missing, insert `k: default` and return it |
| `d.update(other)` / `d.update(k=v)` | merge another mapping or keyword pairs in place |
| `d.pop(k[, default])` | remove key and return value (`default` if missing, else `KeyError`) |
| `d.popitem()` | remove and return the last `(key, value)` pair (LIFO in modern Python) |
| `d.clear()` | remove all items |
| `d.copy()` | shallow copy |
| `d.fromkeys(it, value)` | class method building a dict from keys with one shared value |

`get` is for safe reads; `setdefault` is for "read or initialise" (classic for grouping); `pop` is for "read and remove".
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
d = {"a": 1, "b": 2}

print("setdefault existing:", d.setdefault("a", 99))
print("setdefault new:", d.setdefault("c", 3), "| d:", d)

d.update({"d": 4})
d.update(e=5)
print("after update:", d)

print("pop('b'):", d.pop("b"))
print("pop missing w/ default:", d.pop("zzz", "not there"))
print("popitem:", d.popitem(), "| d:", d)

# fromkeys shares the same value object for every key (watch mutables).
shared = dict.fromkeys(["x", "y"], 0)
print("fromkeys:", shared)
print("copy:", d.copy())
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 13. Dict comprehensions, iterating, merging and nesting

- Dict comprehension: `{k_expr: v_expr for item in iterable if cond}`.
- Iterating a bare dict yields its **keys**; use `.items()` to get `(key, value)` pairs and `.values()` for values.
- Merging: `d1 | d2` (Python 3.9+), `{**d1, **d2}`, or `d1.update(d2)`. Order matters: later mappings win on duplicate keys.
- Nested dicts are dicts whose values are dicts — use chained indexing `d["outer"]["inner"]` or `.get()` at each level to avoid `KeyError`.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
squares = {x: x * x for x in range(1, 6)}
print("dict comprehension:", squares)

d = {"a": 1, "b": 2}
for key in d:
    print("key:", key)
for key, value in d.items():
    print("pair:", key, value)

left = {"a": 1, "shared": "L"}
right = {"b": 2, "shared": "R"}
print("merge operator:", left | right)
print("merge unpacking:", {**left, **right})

users = {"ada": {"role": "admin", "active": True},
         "bob": {"role": "user", "active": False}}
print("nested:", users["ada"]["role"])
print("safe nested:", users.get("cara", {}).get("role", "guest"))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 14. `defaultdict` and `Counter`

Both live in `collections` and subclass `dict`.

- `defaultdict(factory)` calls `factory()` to create a value for a missing key, so `dd[k] += 1` works without an explicit existence check. Use `list` for grouping, `int` for counting, `set` for unique collections.
- `Counter(iterable)` counts hashable items. Useful methods: `most_common(n)`, `elements()`, `subtract(other)`, and it supports `+`, `-`, `&`, `|`. Missing keys return `0` rather than raising.
- Plain dict + `setdefault` can also group, but `defaultdict` is cleaner.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from collections import defaultdict, Counter

# Grouping with defaultdict(list).
groups = defaultdict(list)
for word in ["apple", "ape", "bat", "banana", "bear"]:
    groups[word[0]].append(word)
print("grouped by first letter:", dict(groups))

# Counting with defaultdict(int).
counts = defaultdict(int)
for ch in "abracadabra":
    counts[ch] += 1
print("char counts:", dict(counts))

# Counter.
c = Counter("abracadabra")
print("most common:", c.most_common(3))
print("count of 'z' (missing):", c["z"])
print("total elements:", sum(c.values()))

# Counter arithmetic.
print("c + c:", Counter("aab") + Counter("abc"))
print("intersection:", Counter("aab") & Counter("abc"))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 15. Functional tools: `map`, `filter`, `reduce`, `zip`, `enumerate`

These process whole iterables without explicit loops:

- `map(func, *iterables)` → an **iterator** applying `func` to each item. Use `list(...)` to materialise.
- `filter(predicate, iterable)` → an iterator keeping items where `predicate(item)` is truthy; `filter(None, it)` drops falsy values.
- `functools.reduce(func, iterable[, initializer])` → repeatedly combines items into one value; with an initializer, `func(init, first)` is called first. Empty iterables without an initializer raise `TypeError`.
- `zip(*iterables)` → an iterator of tuples, stopping at the shortest input. `zip(*pairs)` **unzips**. Python 3.10+ accepts `strict=True` to raise if lengths differ.
- `enumerate(iterable, start=0)` → an iterator of `(index, value)` pairs.
- `all(it)` / `any(it)` → `True` if all / any items are truthy (empty → `True` / `False`).
- `sorted(it, key=..., reverse=...)` → new sorted list.

All of these are lazy iterators except `reduce`, `all`, `any` and `sorted`, which consume and return a result.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# map: apply a function to every element.
values = [1, 2, 3, 4]
squared = map(lambda x: x * x, values)
print("map ->", list(squared))
print("map multi-iter:", list(map(lambda x, y: x + y, [1, 2], [10, 20])))
print("map str:", list(map(str, [1, 2, 3])))

# filter: keep elements passing a predicate.
print("filter evens:", list(filter(lambda x: x % 2 == 0, values)))
print("filter None (drops falsy):", list(filter(None, [0, "", 1, "x", None])))
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from functools import reduce

# reduce collapses an iterable to a single value.
print("sum:", reduce(lambda a, b: a + b, [1, 2, 3, 4]))
print("product:", reduce(lambda a, b: a * b, [1, 2, 3, 4]))
print("with initializer:", reduce(lambda a, b: a + b, [1, 2, 3], 100))
print("max via reduce:", reduce(lambda a, b: a if a > b else b, [3, 9, 2]))

# Flatten nested lists with reduce.
flat = reduce(lambda acc, row: acc + row, [[1, 2], [3], [4, 5]], [])
print("flattened:", flat)
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
names = ["ada", "bob", "cara"]
scores = [10, 20, 30]

paired = zip(names, scores)
print("zipped:", list(paired))

pairs = [("a", 1), ("b", 2)]
print("unzipped via zip(*):", list(zip(*pairs)))

# zip stops at the shortest iterable.
print("uneven zip:", list(zip([1, 2, 3], ["x"])))

# enumerate supplies indices.
for index, name in enumerate(names, start=1):
    print(index, name)

# all / any.
print("all truthy?", all([1, True, "x"]))
print("any truthy?", any([0, "", 5]))
print("all empty?", all([]), "any empty?", any([]))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 16. Comprehensions vs `map`/`filter`

For simple transformations the three styles are equivalent:

```python
[x * x for x in xs if x > 0]                 # comprehension
list(map(lambda x: x * x, filter(lambda x: x > 0, xs)))   # map + filter
```

Guidelines:

- **Comprehensions** are usually more readable in Python when both a transform and a filter are involved; they avoid `lambda` and are generally faster for simple cases.
- **`map`/`filter`** shine when you already have a named function, or when you want to pass the lazy iterator onward without materialising a list. With multiple iterables `map(f, a, b)` is handy.
- **`reduce`** is for genuine fold/accumulate operations; for `sum`, `max`, `min` prefer the built-ins.
- Prefer generator expressions `(...)` over list comprehensions for large streams you only iterate once.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
xs = list(range(1, 11))

comp = [x * x for x in xs if x % 2 == 0]
func = list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, xs)))
print("comprehension:", comp)
print("map+filter:   ", func)
print("same?", comp == func)

# A named function makes map cleaner than a lambda.
def celsius(f):
    return (f - 32) * 5 / 9

print("celsius:", list(map(celsius, [32, 68, 100])))

# Generator expression: lazy, memory-friendly.
gen = (x * x for x in xs)
print("generator type:", type(gen).__name__, "| sum:", sum(gen))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 17. Modules — what, why and import forms

A **module** is simply a `.py` file. It can define functions, classes and variables, and it has its own namespace. Modules let you organise code into reusable files, avoid name collisions, and share code across programs.

Import forms:

| Form | Example | Effect |
|---|---|---|
| `import m` | `import math` | bind module name; access via `math.sqrt` |
| `import m as alias` | `import datetime as dt` | bind an alias |
| `from m import name` | `from math import sqrt` | bind the name directly |
| `from m import name as alias` | `from math import sqrt as root` | bind under an alias |
| `from m import *` | — | imports public names (discouraged: pollutes namespace) |
| `import pkg.mod` | `import os.path` | import a submodule |

Only one copy of a module is loaded per session; `sys.modules` is the cache. `importlib.reload(module)` re-executes a module after its file changes.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
import math as m
from math import sqrt, pi
from math import factorial as fact

print("alias:", m.sqrt(16))
print("direct:", sqrt(9), "pi =", round(pi, 4))
print("fact(5):", fact(5))
print("modules cache has math?", "math" in sys.modules)
print("reload math ->", importlib.reload(math).__name__)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 18. `__name__ == "__main__"` and the module search path

When the interpreter runs a file, it sets that file's module `__name__` to `"__main__"`. When the file is *imported*, `__name__` is the module's name. The idiom

```python
if __name__ == "__main__":
    main()
```

runs code only when the file is executed directly, not when imported — this makes a file both a script and a reusable module.

The **module search path** is `sys.path`: a list of directories searched on `import`. It includes the script's directory, `PYTHONPATH` entries, and standard-library/site-packages locations. To import from another folder you can append to `sys.path` (as we do below for a user module). `sys.path[0]` is normally the current script's directory.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
print("this cell's __name__ =", __name__)

# The first entries of sys.path are searched first ('' means current dir).
for entry in sys.path[:4]:
    print("sys.path entry:", repr(entry))

print("sys.version:", sys.version.split()[0])
print("current working dir:", os.getcwd())
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 19. Standard-library tour — `math`, `random`, `statistics`, `datetime`

- **math**: `sqrt`, `floor`, `ceil`, `pow`, `log`, `gcd`, `factorial`, `isclose`, constants `pi`, `e`, `inf`, `nan`.
- **random**: `seed`, `random()` (float in [0,1)), `randint(a,b)` (inclusive), `randrange`, `choice`, `choices` (with replacement + weights), `sample` (without replacement), `shuffle` (in place), `uniform`, `gauss`.
- **statistics**: `mean`, `median`, `mode`, `stdev`, `variance`, `quantiles`.
- **datetime**: classes `date`, `time`, `datetime`, `timedelta`, `timezone`; `datetime.now()`, `date.today()`, `strftime`/`strptime` for formatting/parsing, and timedelta arithmetic.

Seeding `random` makes results reproducible — essential for teaching examples and tests.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# math
print("sqrt(81):", math.sqrt(81), "floor/ceil:", math.floor(2.7), math.ceil(2.2))
print("gcd(12, 18):", math.gcd(12, 18), "factorial(6):", math.factorial(6))
print("pi:", round(math.pi, 4), "isclose:", math.isclose(0.1 + 0.2, 0.3))

# random (seeded for reproducibility)
random.seed(42)
print("random():", round(random.random(), 4))
print("randint(1,6):", random.randint(1, 6))
print("choice:", random.choice(["a", "b", "c"]))
print("sample:", random.sample(range(10), 3))
deck = [1, 2, 3, 4]
random.shuffle(deck)
print("shuffle:", deck)

# statistics
data = [1, 2, 2, 3, 4, 7, 9]
print("mean:", statistics.mean(data), "median:", statistics.median(data))
print("mode:", statistics.mode(data), "stdev:", round(statistics.stdev(data), 3))

# datetime
today = dt.date.today()
now = dt.datetime.now()
delta = dt.timedelta(days=7)
print("today:", today, "now:", now.strftime("%Y-%m-%d %H:%M"))
print("one week later:", (now + delta).date())
print("parsed:", dt.datetime.strptime("2024-01-15", "%Y-%m-%d").date())
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 20. Standard-library tour — `collections`, `itertools`, `functools`

- **collections**: `Counter`, `defaultdict`, `deque` (fast append/pop at both ends), `namedtuple` (tuple with named fields), `OrderedDict`, `ChainMap`.
- **itertools** — lazy iterators for combinatorics and streaming:
  - `chain(*its)` flatten iterables; `product` cartesian product; `permutations`; `combinations`; `combinations_with_replacement`.
  - `count`, `cycle`, `repeat` — infinite iterators (use `islice` to cap them).
  - `accumulate`, `groupby`, `islice`, `zip_longest`, `filterfalse`, `takewhile`, `dropwhile`, `starmap`, `compress`; `pairwise` (3.10+), `batched` (3.12+).
- **functools**: `reduce`, `lru_cache` (memoisation), `partial` (fix arguments), `wraps` (for decorators), `singledispatch`.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from collections import deque, namedtuple
from itertools import chain, product, combinations, count, islice, accumulate, groupby
from functools import lru_cache, partial

# collections
dq = deque([1, 2, 3])
dq.appendleft(0); dq.append(4)
print("deque:", dq, "pop:", dq.pop(), dq.popleft())

Point = namedtuple("Point", "x y")
p = Point(3, 4)
print("namedtuple:", p, p.x, p.y)

# itertools
print("chain:", list(chain([1, 2], [3, 4])))
print("product:", list(product([1, 2], ["a", "b"])))
print("combinations:", list(combinations([1, 2, 3], 2)))
print("first 5 of count:", list(islice(count(10, 2), 5)))
print("accumulate:", list(accumulate([1, 2, 3, 4])))

for key, group in groupby("AAABB"):
    print("groupby:", key, list(group))

# functools
@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print("fib(20):", fib(20), "cache:", fib.cache_info())
add5 = partial(lambda a, b: a + b, 5)
print("partial:", add5(10))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 21. Standard-library tour — `os`, `sys`, `json`, `csv`, `re`, `pathlib`

- **os**: `os.getcwd()`, `os.listdir()`, `os.mkdir`/`makedirs`, `os.path.join`, `os.environ`, `os.sep`.
- **sys**: `sys.path`, `sys.argv`, `sys.version`, `sys.exit()`.
- **json**: `json.dumps(obj)` (object → string), `json.loads(s)` (string → object), plus `dump`/`load` for files. Handles dict, list, str, number, bool, `None`.
- **csv**: `csv.reader`, `csv.writer`, `csv.DictReader`, `csv.DictWriter`; use `newline=""` when opening files.
- **re**: `re.search`, `re.match`, `re.findall`, `re.sub`, `re.split`, `re.compile` for reuse.
- **pathlib**: `Path` objects with `exists()`, `is_file()`, `is_dir()`, `mkdir()`, `glob()`, `read_text()`, `write_text()`, and operators like `p / "child"`, `p.parent`, `p.stem`, `p.suffix`.

We keep the JSON/CSV demos in memory with `io.StringIO` so this notebook runs without external files.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# os / sys already imported above.
print("cwd:", os.getcwd(), "sep:", repr(os.sep))
print("argv count:", len(sys.argv), "version:", sys.version.split()[0])

# json: round-trip a nested structure.
payload = {"name": "Ada", "scores": [10, 20], "active": True}
text = json.dumps(payload, sort_keys=True)
print("json.dumps:", text)
print("json.loads:", json.loads(text))

# csv: read/write in memory with StringIO.
buffer = io.StringIO()
writer = csv.writer(buffer)
writer.writerow(["name", "score"])
writer.writerow(["ada", 10])
csv_text = buffer.getvalue()
print("csv text:", repr(csv_text))
print("csv.reader:", list(csv.reader(io.StringIO(csv_text))))

# re: patterns.
email = "Contact ada@example.com or bob@test.org"
print("findall emails:", re.findall(r"[\w.]+@[\w.]+", email))
print("sub:", re.sub(r"\d+", "#", "abc 123 def 45"))

# pathlib: build and inspect a path (no file needed).
p = DATA / "iris.csv"
print("path:", p, "| parent:", p.parent, "| stem:", p.stem, "| suffix:", p.suffix)
print("data dir exists?", DATA.exists())
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 22. Creating and importing a user-defined module

We can write a real `.py` module from inside the notebook using `Path.write_text`, add its folder to `sys.path`, and import it like any other module. `importlib.reload` makes re-running this cell pick up changes.

The module below defines a constant and two functions — the same pattern you would use for your own reusable data-science helpers.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# Write a user-defined module to disk, then import it.
module_dir = Path("_ds_modules")
module_dir.mkdir(exist_ok=True)

module_source = '''\
'A tiny user-defined module created for this lesson.'
PI = 3.14159

def mean(values):
    'Arithmetic mean of a sequence of numbers.'
    return sum(values) / len(values)

def summarize(values):
    'Return count, mean, min and max in a dictionary.'
    return {"n": len(values), "mean": mean(values),
            "min": min(values), "max": max(values)}
'''
(module_dir / "stats_utils.py").write_text(module_source, encoding="utf-8")

# Make the folder importable for this session.
module_path = str(module_dir.resolve())
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import stats_utils
importlib.reload(stats_utils)

print("module name:", stats_utils.__name__)
print("PI:", stats_utils.PI)
print("mean:", stats_utils.mean([1, 2, 3, 4]))
print("summarize:", stats_utils.summarize([1, 2, 3, 4]))
print("file:", stats_utils.__file__)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 23. Packages and `__init__.py`

A **package** is a folder of modules. A regular package contains an `__init__.py` file (which may be empty) that marks the folder as importable and can re-export names. A submodule is imported as `package.module`, or its names can be lifted into the package namespace by `__init__.py`.

Example layout:

```
mypkg/
    __init__.py     -> from .core import greet
    core.py         -> def greet(name): ...
```

Then `from mypkg import greet` works. We create that structure below. (Python 3.3+ also supports *namespace packages* without `__init__.py`, but regular packages are the common case.)
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# Build a small package inside our module directory.
pkg_dir = module_dir / "mypkg"
pkg_dir.mkdir(exist_ok=True)
(pkg_dir / "__init__.py").write_text(
    'from .core import greet\n', encoding="utf-8")
(pkg_dir / "core.py").write_text(
    'def greet(name):\n    return f"Hello, {name}!"\n', encoding="utf-8")

from mypkg import greet
print("package function:", greet("world"))
print("package file:", sys.modules["mypkg"].__file__)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## 24. Practical data-science examples

Two tasks that recur constantly:

1. **Counting words** — `Counter` gives frequencies directly; `most_common(n)` ranks them.
2. **Grouping records** — `defaultdict(list)` buckets records by a key.

We also use `sorted(..., key=...)` to order results and a dict comprehension to reshape a mapping.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from collections import Counter, defaultdict

# 1) Counting words in a short text.
text = "data science is fun and data is everywhere in science data"
words = re.findall(r"\w+", text.lower())
word_counts = Counter(words)
print("top 3 words:", word_counts.most_common(3))
print("'data' count:", word_counts["data"])

# 2) Grouping records by category.
records = [
    {"dept": "sales", "name": "Ada", "salary": 70},
    {"dept": "eng",   "name": "Bob", "salary": 90},
    {"dept": "sales", "name": "Cara", "salary": 80},
]
by_dept = defaultdict(list)
for rec in records:
    by_dept[rec["dept"]].append(rec)

for dept, members in sorted(by_dept.items()):
    avg = statistics.mean(m["salary"] for m in members)
    print(dept, "->", [m["name"] for m in members], "avg salary:", avg)

# Reshape: name -> salary via dict comprehension.
name_to_salary = {r["name"]: r["salary"] for r in records}
print("name -> salary:", name_to_salary)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Try it yourself 1 — list & set practice

Given `scores = [5, 3, 9, 3, 7, 5, 3]`:

1. Use a **list comprehension** to keep only the even scores.
2. Use `set` to find the unique scores, then print them sorted.
3. Use `sorted` with a `key` to sort the original list by distance from 5 (closest first).

Write your code in the cell below, then compare with the solution.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# YOUR CODE HERE
scores = [5, 3, 9, 3, 7, 5, 3]
# 1. even scores via comprehension
# 2. unique scores sorted
# 3. sorted by distance from 5
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
### Solution 1
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
scores = [5, 3, 9, 3, 7, 5, 3]

even_scores = [x for x in scores if x % 2 == 0]
print("even scores:", even_scores)

unique_sorted = sorted(set(scores))
print("unique sorted:", unique_sorted)

by_distance = sorted(scores, key=lambda x: abs(x - 5))
print("by distance from 5:", by_distance)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Try it yourself 2 — functional tools

Given `names = ["ada", "bob", "cara", "dan"]` and `ages = [36, 25, 41, 30]`:

1. Pair each name with its age using `zip`, then print the pairs.
2. Use `map` with `str.upper` to get uppercase names.
3. Use `filter` to keep names longer than 3 characters.
4. Use `reduce` to concatenate all names into one comma-separated string.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# YOUR CODE HERE
names = ["ada", "bob", "cara", "dan"]
ages = [36, 25, 41, 30]
# 1. pairs via zip
# 2. uppercase via map
# 3. names longer than 3 via filter
# 4. concatenate via reduce
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
### Solution 2
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from functools import reduce

names = ["ada", "bob", "cara", "dan"]
ages = [36, 25, 41, 30]

print("pairs:", list(zip(names, ages)))
print("uppercase:", list(map(str.upper, names)))
print("long names:", list(filter(lambda n: len(n) > 3, names)))
print("joined:", reduce(lambda a, b: a + ", " + b, names))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Try it yourself 3 — dictionaries & modules

1. Build a dict comprehension mapping each number 1–5 to its cube.
2. Use a `defaultdict(list)` to group words by their length from the sentence `"the quick brown fox jumps"`.
3. Call `stats_utils.summarize` (the module you created above) on the list of word lengths.
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
# YOUR CODE HERE
# 1. cubes dict comprehension
# 2. group words by length
# 3. summarize the lengths using the user-defined module
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
### Solution 3
"""))

# ---------------------------------------------------------------------------
cells.append(code(r"""
from collections import defaultdict

cubes = {n: n ** 3 for n in range(1, 6)}
print("cubes:", cubes)

words = "the quick brown fox jumps".split()
by_length = defaultdict(list)
for word in words:
    by_length[len(word)].append(word)
print("by length:", dict(by_length))

lengths = [len(w) for w in words]
print("summarize lengths:", stats_utils.summarize(lengths))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Cheat sheet / key takeaways

| Topic | Essentials |
|---|---|
| Sequences | `str`, `list`, `tuple`, `range` — ordered, indexable, sliceable with `x[i:j:k]`. |
| Mutable vs immutable | Mutable: `list`, `set`, `dict`. Immutable/hashable: `str`, `tuple`, `int`, `frozenset`. |
| List creation | `[]`, `list(it)`, `[expr for x in it]`. Index `L[-1]` = last. `L[::-1]` = reversed copy. |
| List methods | `append`, `extend`, `insert`, `remove`, `pop`, `clear`, `index`, `count`, `sort`, `reverse`, `copy` (in-place ones return `None`). |
| Sorting | `L.sort()` in place; `sorted(L, key=..., reverse=...)` new list; stable. |
| Copying | `b = a` aliases; `.copy()` shallow; `copy.deepcopy()` deep. |
| Comprehension | `[expr for x in it if cond]`; nested `for`s; `[[...] for i in ...]` for matrices. |
| Tuple | `(1, 2)`, single `(5,)`, immutable, hashable, methods `count`/`index`, unpack `a, *rest = t`. |
| Set | `{1, 2}`, `set(it)`; `add`/`discard`/`pop`; `\| & - ^`; subset `<=`; `frozenset` hashable. |
| Dedup | `list(set(xs))` (unordered) vs `list(dict.fromkeys(xs))` (ordered). |
| Dict | `{k: v}`, `dict(...)`; `get`, `setdefault`, `update`, `pop`, `popitem`, `keys`/`values`/`items`; insertion-ordered since 3.7; merge `d1 \| d2`. |
| Special dicts | `defaultdict(list/int/set)`, `Counter` with `most_common`. |
| Functional | `map(f, it)`, `filter(pred, it)`, `reduce(f, it, init)`, `zip(*its)`, `enumerate(it)`, `all`/`any`, `sorted(key=)`. |
| Modules | `import m`, `import m as x`, `from m import n`; `__name__ == "__main__"`; `sys.path`; `importlib.reload`. |
| Std lib | `math`, `random`, `statistics`, `datetime`, `collections`, `itertools`, `functools`, `os`, `sys`, `json`, `csv`, `re`, `pathlib`. |
| Packages | Folder + `__init__.py`; `import pkg.mod`; namespace packages need no `__init__.py`. |

Key habits: prefer comprehensions for transform+filter, `get`/`setdefault` for safe dict access, `deepcopy` for nested structures, and `__name__` guards to make files both scripts and modules.
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""
## Further reading

- [Python 3 tutorial — More Control Flow Tools (functions, `lambda`)](https://docs.python.org/3/tutorial/controlflow.html)
- [Python 3 — `itertools` recipes](https://docs.python.org/3/library/itertools.html#itertools-recipes)
- [Python 3 — `functools` reference](https://docs.python.org/3/library/functools.html)
- [Python 3 — `collections` — container datatypes](https://docs.python.org/3/library/collections.html)
- [Python 3 — `copy` module (shallow vs deep copy)](https://docs.python.org/3/library/copy.html)
- [Python 3 — `importlib` (reload, import machinery)](https://docs.python.org/3/library/importlib.html)
- [Python 3 — `sys` module (`sys.path`, `sys.modules`)](https://docs.python.org/3/library/sys.html)
- [Python 3 — `re` — regular expression operations](https://docs.python.org/3/library/re.html)
- [Python 3 — `datetime` — basic date and time types](https://docs.python.org/3/library/datetime.html)
- [The Python Standard Library (full index)](https://docs.python.org/3/library/index.html)
"""))

target = Path(__file__).resolve().parents[1] / "notebooks" / "python-fundamentals" / "unit2_data_structures.ipynb"
write_notebook(target, cells)
