"""Builder for Unit I - Fundamentals of Python Programming-1 notebook."""

from nbhelper import md, code, write_notebook

OUT = "notebooks/python-fundamentals/unit1_python_fundamentals.ipynb"

cells = []

# 1 - Title / intro / objectives
cells.append(md(r'''
# Unit I — Fundamentals of Python Programming-1

Python is the general-purpose language behind most modern data-science tooling. This notebook is a lasting reference for the **core language**: its history and features, data types, variables and assignment, input/output, operators and expressions, strings, conditional statements, loops and loop control, and functions (built-in, user-defined, recursive, and lambda). Every code cell is small, runnable, and commented so you can re-run or adapt it later. The final sections add three practice exercises, a small data-science example, and compact cheat sheets.

## Learning objectives
By the end of this unit you should be able to:

- Describe Python's origins, key features, and where it is used, especially in data science.
- Install/check Python, run cells in Jupyter, and use basic notebook operations.
- Name the core data types (`int`, `float`, `complex`, `bool`, `str`, `bytes`, `NoneType`) and inspect them with `type()` and `isinstance()`.
- Declare variables, apply naming rules, and use chained, unpacking, starred, and augmented assignment.
- Produce formatted output with `print()`, `sep`, `end`, f-strings, and format specs, and read input with `input()`.
- Use every major operator family (arithmetic, comparison, logical, bitwise, assignment, identity, membership) and predict evaluation with precedence rules.
- Build and evaluate expressions, understanding associativity and evaluation order.
- Index and slice strings and apply the common string methods; write escape sequences, raw strings, and f-strings.
- Control flow with `if`/`elif`/`else`, nested conditionals, the ternary expression, and truthiness rules.
- Iterate with `for` and `while`, `range()`, `enumerate()`, `zip()`, and nested loops; steer loops with `break`, `continue`, `pass`, and the loop `else` clause.
- Use the important built-in functions and write user-defined functions with positional, keyword, default, `*args`, and `**kwargs` parameters, `return`, docstrings, and correct scope (`global`/`nonlocal`).
- Write recursive functions with a proper base case and explain when recursion is appropriate.
- Write lambda functions and use them with higher-order functions such as `map()`, `filter()`, and `sorted()`.
'''))

# 2 - Official documentation
cells.append(md(r'''
## Official documentation & references

These are the authoritative sources used throughout this notebook. Bookmark them; they are the definitive reference for current Python versions.

- [Python 3 Tutorial — Introduction](https://docs.python.org/3/tutorial/introduction.html)
- [Python 3 Tutorial — Control flow tools](https://docs.python.org/3/tutorial/controlflow.html)
- [Python 3 Standard Library — Built-in functions](https://docs.python.org/3/library/functions.html)
- [Python 3 Standard Library — Built-in types](https://docs.python.org/3/library/stdtypes.html)
- [Python 3 Tutorial — Input and Output](https://docs.python.org/3/tutorial/inputoutput.html)
- [Python 3 Tutorial — Data structures](https://docs.python.org/3/tutorial/datastructures.html)
- [About Python](https://www.python.org/about/)
- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
'''))

# 3 - First code cell: imports and setup
cells.append(code(r'''
from pathlib import Path
DATA = Path("../../data")

import sys
import math
import csv
import io

print("Python", sys.version.split()[0])
print("Executable:", sys.executable)
'''))

# 4 - History, features, uses
cells.append(md(r'''
## 1. Python: history, features, and uses

**History.** Python was created by **Guido van Rossum** at CWI in the Netherlands and first released in **1991** as a successor to the ABC language. The name comes from the comedy group *Monty Python*, not the snake. Python 2 (2000) dominated for a decade; Python 3 (2008) is the current line, with type hints, f-strings, and many clarity improvements added over time.

**Features.**

- **Interpreted and interactive** — code runs line by line, ideal for notebooks and REPL experimentation.
- **High-level and readable** — English-like syntax and significant indentation instead of braces.
- **Dynamically typed** — variables get a type from the value assigned; no declaration required.
- **Multi-paradigm** — procedural, object-oriented, and functional styles all work.
- **Batteries included** — a large standard library plus an enormous third-party ecosystem (PyPI).
- **Portable and extensible** — runs on Windows/macOS/Linux and can call C/C++ libraries.
- **Memory managed** — automatic garbage collection; you rarely manage memory by hand.

**Uses.** Web back ends (Django, Flask, FastAPI), data analysis and visualisation (NumPy, pandas, Matplotlib, Seaborn), machine learning and deep learning (scikit-learn, XGBoost, PyTorch, TensorFlow), scientific computing (SciPy, SymPy), automation and scripting, DevOps tooling, NLP, finance, and education. In data science Python is the glue that loads data, cleans it, models it, and communicates results — which is why this whole learning environment is built around it.

**Where Python fits in a DS workflow:** load (`pandas.read_csv`) → explore (DataFrame methods, statistics) → visualise (Matplotlib/Seaborn) → model (scikit-learn) → evaluate and communicate. This unit covers the language foundations every later step assumes.
'''))

# 5 - small feature demo
cells.append(code(r'''
# Python as a calculator, with readable high-level syntax
radius = 5
area = math.pi * radius ** 2
print(f"Area of circle with r={radius}: {area:.2f}")

message = "data science"
print(message.title(), "->", len(message), "characters")
'''))

# 6 - Installation & Jupyter basics
cells.append(md(r'''
## 2. Installation and Jupyter basics

**Installation.** Download Python from [python.org](https://www.python.org/downloads/) or install it with your OS package manager. This project uses a **uv-managed virtual environment** at `.venv` with all data-science packages preinstalled. Check an install with `python3 --version` and `python3 -m pip --version`. A virtual environment isolates project packages from the system Python.

**Jupyter.** Launch the notebook server with `jupyter lab` (or `jupyter notebook`) and open a `.ipynb` file. A notebook is a list of **cells**:

- **Markdown cells** hold formatted text (like this one); render them with `Shift+Enter`.
- **Code cells** hold Python; run them with `Shift+Enter`. The output appears below the cell.
- The **kernel** is the Python process executing the code — here it is named `python3`.

Useful notebook features:

- `Tab` completion and `Shift+Tab` for a function's signature/docstring.
- `?object` or `help(object)` opens documentation.
- **Magics** start with `%` (line) or `%%` (cell), e.g. `%time`, `%matplotlib inline`, `%pwd`.
- `Ctrl+Enter` runs a cell without moving; `Run All` executes top to bottom.

The cells below are plain Python so they also run outside Jupyter.
'''))

# 7 - environment demo
cells.append(code(r'''
# Notebook/kernel information
print(sys.version_info)
print("Executable:", sys.executable)
print("pi from the math module:", math.pi)

# Notebook-only conveniences, shown as comments so the cell runs in any Python:
# help(len)          # documentation popup in Jupyter
# %time sum(range(1_000_000))
# %matplotlib inline
'''))

# 8 - Literals and data types
cells.append(md(r'''
## 3. Literals and data types

A **literal** is a fixed value written directly in source code. Python's built-in scalar types:

| Type | Example literals | Notes |
|---|---|---|
| `int` | `42`, `-7`, `0b1010`, `0o17`, `0xFF`, `1_000_000` | Arbitrary precision; underscores aid readability |
| `float` | `3.14`, `1e-3`, `-2.0`, `float("inf")` | IEEE-754 double precision |
| `complex` | `2+3j`, `complex(1, -2)` | `.real` and `.imag` attributes |
| `bool` | `True`, `False` | Subclass of `int`: `True == 1` |
| `str` | `"hello"`, `'world'`, triple-quoted multi-line | Immutable sequence of Unicode characters |
| `bytes` | `b"ABC"` | Immutable sequence of bytes (0–255) |
| `NoneType` | `None` | The single `None` value marks "no value" |

Use `type(x)` for the exact type and `isinstance(x, T)` for type checking — the latter accepts a tuple of types and respects inheritance (so `isinstance(True, int)` is `True`). Collections such as `list`, `tuple`, `dict`, and `set` are covered through the built-ins later. Most of these types are **immutable**; reassignment, not mutation, changes a variable bound to them.
'''))

# 9 - types demo
cells.append(code(r'''
values = [42, 3.14, 2 + 3j, True, "hello", b"ABC", None]
for value in values:
    print(f"{str(value):>8}  ->  {type(value).__name__}")

print("bool is an int subclass:", isinstance(True, int))
print("isinstance with a tuple:", isinstance(3.5, (int, float)))
print("complex parts:", (2 + 3j).real, (2 + 3j).imag)

# Numeric literal forms all evaluate to plain integers/floats
print(0b1010, 0o17, 0xFF, 1_000_000, 1.5e2)
print("None is unique:", None is None)
'''))

# 10 - Variables & assignment
cells.append(md(r'''
## 4. Variables and assignment

A **variable** is a name bound to an object. Assignment `name = value` evaluates the right-hand side first, then binds the name. No type declaration is needed.

**Naming rules**

- May contain letters, digits, and underscores; must not start with a digit.
- Cannot be a Python keyword (`if`, `for`, `class`, `None`, …).
- Case-sensitive: `score` and `Score` are different.
- Convention (PEP 8): `snake_case` for variables/functions, `UPPER_CASE` for module-level **constants** (a convention, not enforced), leading underscore for "internal" names.

**Assignment forms**

- Chained: `a = b = c = 0` binds all three names to the same object.
- Multiple/unpacking: `x, y = 1, 2` pairs names with values (the right side is iterable).
- Starred unpacking: `first, *rest = [1, 2, 3, 4]` collects the remainder into a list.
- Swapping: `a, b = b, a` works without a temporary variable.
- Augmented: `x += 1`, `x *= 2`, `x //= 3`, etc. — shorthand for `x = x + 1`.
'''))

# 11 - assignment demo
cells.append(code(r'''
# Chained, unpacking, starred, swapping, augmented assignment
a = b = c = 0
print("chained:", a, b, c)

x, y = 1, 2
x, y = y, x
print("swapped:", x, y)

first, *middle, last = [1, 2, 3, 4, 5]
print("starred:", first, middle, last)

score = 10
score += 5      # 15
score *= 2      # 30
score //= 4     # 7
print("augmented result:", score)

MAX_RETRIES = 3   # constant by convention only
print("constant:", MAX_RETRIES)
'''))

# 12 - I/O
cells.append(md(r'''
## 5. Input and output

**`print(*objects, sep=' ', end='\n', file=sys.stdout)`** converts each object to a string, joins them with `sep`, and finishes with `end`.

**Formatted strings**

- **f-strings** (Python 3.6+): `f"{name} scored {score:.1f}"` — expressions inside braces, format specs after `:`.
- Common format specs: `:.2f` (2 decimals), `:>10` (right-align width 10), `:<10` / `:^10` (left/centre), `:,.2f` (thousands separator), `:.0%` (percentage), `:d` (integer), `:x` (hex).
- `str.format()` (e.g. `"{} {}".format(a, b)`) and `%`-formatting (`"%s %d" % (a, b)`) are older but still seen.
- `!r` in an f-string uses `repr()` instead of `str()` — handy for debugging: `f"{name!r}"`.

**`input(prompt)`** prints a prompt and returns whatever the user types as a **`str`**. Convert it explicitly with `int()`, `float()`, etc. It is interactive, so this notebook stands in for the typed value with a string variable.
'''))

# 13 - print/input demo
cells.append(code(r'''
name = "Ada"
score = 92.5

print("Name:", name, "Score:", score)
print("Name:", name, "Score:", score, sep=" | ", end="!\n")

# f-strings and format specifications
print(f"{name:>10} scored {score:5.1f} ({score / 100:.0%})")
print(f"{1234567.891:,.2f} | {42:04d} | {255:#x} | {name!r}")

# Older formatting styles you may encounter
print("{} has {}".format(name, int(score)))
print("%s has %d" % (name, int(score)))

# input() returns a string; conversion is explicit
raw = "42"                      # stand-in for: raw = input("Enter an integer: ")
value = int(raw)
print(type(raw).__name__, "->", type(value).__name__, value + 1)
'''))

# 14 - arithmetic operators
cells.append(md(r'''
## 6. Operators I — arithmetic

| Operator | Meaning | Example |
|---|---|---|
| `+` `-` | addition, subtraction | `7 + 2`, `7 - 2` |
| `*` | multiplication | `7 * 2` |
| `/` | true division → `float` | `7 / 2` → `3.5` |
| `//` | floor division | `7 // 2` → `3`, `-7 // 2` → `-4` |
| `%` | remainder (modulo) | `7 % 2` → `1` |
| `**` | exponentiation | `2 ** 3` → `8` |
| `+x` `-x` | unary plus/minus | `-5` |
| `divmod(a, b)` | returns `(a // b, a % b)` | `divmod(7, 2)` → `(3, 1)` |
| `pow(a, b, m)` | modular exponentiation | `pow(2, 3, 5)` → `3` |

`**` binds more tightly than unary minus on the left (`-2 ** 2` is `-4`), and is **right-associative** (`2 ** 3 ** 2` is `512`). Mixed `int`/`float` arithmetic produces a `float`; division by zero raises `ZeroDivisionError`.
'''))

# 15 - arithmetic demo
cells.append(code(r'''
print(7 + 2, 7 - 2, 7 * 2)
print("true division:", 7 / 2)
print("floor division:", 7 // 2, -7 // 2)
print("modulo:", 7 % 2, -7 % 2)
print("power:", 2 ** 3, "right-associative:", 2 ** 3 ** 2)
print("unary vs power:", -2 ** 2, "vs", (-2) ** 2)
print("divmod:", divmod(7, 2))
print("pow with modulus:", pow(2, 3, 5))
'''))

# 16 - comparison and logical
cells.append(md(r'''
## 7. Operators II — comparison and logical

**Comparison** operators return `bool`: `==`, `!=`, `<`, `>`, `<=`, `>=`. Chained comparisons are evaluated naturally: `1 < x < 10` means `1 < x and x < 10`, with `x` evaluated once.

**Logical** operators: `and`, `or`, `not`. They **short-circuit** (stop as soon as the result is known) and return one of the *operands*, not necessarily a `bool`:

- `a and b` → returns `a` if `a` is falsy, otherwise `b`.
- `a or b` → returns `a` if `a` is truthy, otherwise `b`.
- `not a` → always returns a `bool`.

Short-circuiting makes idioms like `x = user_input or "default"` and `if obj is not None and obj.ready` safe and efficient.
'''))

# 17 - comparison/logical demo
cells.append(code(r'''
x = 5
print("chained:", 1 < x < 10)
print(3 == 3.0, 3 != 4, x >= 5, x < 3)

print("and/or return operands:", 0 and 7, 3 and 7, "" or "default", 2 or 3)
print("not:", not True, not 0, not [])

def expensive():
    print("  expensive() called")
    return True
print("short-circuit result:", False and expensive())
'''))

# 18 - bitwise, identity, membership
cells.append(md(r'''
## 8. Operators III — bitwise, identity, membership

**Bitwise** operators act on integers at the binary level:

| Op | Meaning | Op | Meaning |
|---|---|---|---|
| `&` | bitwise AND | `|` | bitwise OR |
| `^` | bitwise XOR | `~` | bitwise NOT (invert) |
| `<<` | left shift | `>>` | right shift |

They also work on `set` for `&` (intersection), `|` (union), and `^` (symmetric difference).

**Identity** `is` / `is not` check whether two names reference the *same object*; `==` checks *value equality*. Small integers and interned strings are often shared, so use `is` only for singletons such as `None` (`if x is None:`).

**Membership** `in` / `not in` test containment in a sequence, set, dict (keys), or substring: `3 in [1,2,3]`, `"a" in "cat"`, `"key" in d`.
'''))

# 19 - bitwise/identity/membership demo
cells.append(code(r'''
a, b = 5, 3           # 0b101 and 0b011
print("& | ^ ~:", a & b, a | b, a ^ b, ~a)
print("shifts:", 1 << 4, 16 >> 2)

print("set bitwise:", {1, 2, 3} & {3, 4}, {1, 2} | {2, 3}, {1, 2} ^ {2, 3})

empty1, empty2 = [], []
print("== vs is:", empty1 == empty2, empty1 is empty2)
print("None check:", None is None)

print("membership:", 3 in [1, 2, 3], "a" in "cat", "x" not in "cat", "k" in {"k": 1})
'''))

# 20 - assignment ops, precedence, evaluation order
cells.append(md(r'''
## 9. Assignment operators, precedence, and expression evaluation

**Augmented assignment** operators combine an operation with assignment: `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `<<=`, `>>=`. They rebind the name (with in-place behaviour for mutable objects).

**Precedence**, from lowest to highest (use parentheses whenever in doubt):

| Level | Operators |
|---|---|
| lowest | `lambda`, `if – else` (conditional expression) |
| | `or` |
| | `and` |
| | `not` |
| | comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`, `is`, `in` |
| | `|` |
| | `^` |
| | `&` |
| | `<<`, `>>` |
| | `+`, `-` |
| | `*`, `/`, `//`, `%` |
| | unary `+`, `-`, `~` |
| | `**` (right-associative) |
| highest | call, subscript, attribute access `( ) [ ] .` |

**Evaluation order.** Python generally evaluates expressions **left to right**, and function arguments left to right. Operators of equal precedence associate left-to-right (except `**`, which is right-to-left). Parentheses always override precedence. Understanding this lets you predict `2 + 3 * 4 ** 2` → `50` (power, then multiply, then add).
'''))

# 21 - precedence demo
cells.append(code(r'''
x = 10
x += 3
x **= 2
print("x after += 3 and **= 2:", x)

a = b = 7
print("chained assignment:", a, b)

print("2 + 3 * 4 ** 2 =", 2 + 3 * 4 ** 2)
print("(2 + 3) * 4 ** 2 =", (2 + 3) * 4 ** 2)
print("not 3 > 2        =", not 3 > 2)
print("1 + 2 < 4 and 5 > 3 =", 1 + 2 < 4 and 5 > 3)
print("comparison chaining:", 1 < 2 < 3, 3 < 2 < 1)
'''))

# 22 - Strings indexing/slicing
cells.append(md(r'''
## 10. Strings — indexing and slicing

A string is an **immutable ordered sequence** of characters. Each character has a positive index from `0` and a negative index from `-1` (the last character).

- Indexing: `s[0]`, `s[-1]`.
- Slicing: `s[start:stop:step]` — includes `start`, excludes `stop`; both are optional.
  - `s[:6]` first six characters, `s[6:]` from index 6 onward, `s[::2]` every second character.
  - `s[::-1]` reverses the string.
- Out-of-range **indices** raise `IndexError`, but out-of-range **slice boundaries** are clamped silently.
- Strings cannot be changed in place (`s[0] = "J"` raises `TypeError`); build a new string instead.
- Sequence operations: concatenation `+`, repetition `*`, and membership `in`.
'''))

# 23 - indexing/slicing demo
cells.append(code(r'''
s = "PythonProgramming"
print("length:", len(s))
print("first/last:", s[0], s[-1])
print("slice [6:]:", s[6:])
print("slice [:6]:", s[:6])
print("slice [2:9:2]:", s[2:9:2])
print("reversed:", s[::-1])

print("concat/repeat:", "Py" + "thon", "ha" * 3)
print("membership:", "Pro" in s)
print("clamped slice:", s[:100])
# s[0] = "J"        # TypeError: strings are immutable
'''))

# 24 - string methods
cells.append(md(r'''
## 11. String methods

Strings provide many non-mutating methods (they **return new strings**):

**Case:** `upper()`, `lower()`, `title()`, `capitalize()`, `swapcase()`.

**Whitespace:** `strip()`, `lstrip()`, `rstrip()`; `strip(chars)` removes given characters.

**Search and replace:** `find(sub)` / `rfind(sub)` (index or `-1`), `index(sub)` (raises `ValueError` if missing), `count(sub)`, `replace(old, new[, count])`, `startswith(prefix)`, `endswith(suffix)`.

**Split and join:** `split(sep=None)` (splits on whitespace by default), `rsplit()`, `splitlines()`, and `"sep".join(iterable)` (the inverse of split).

**Alignment/padding:** `center(w)`, `ljust(w)`, `rjust(w)`, `zfill(w)`, `format()` / `format_map()`.

**Predicates (return bool):** `isdigit()`, `isdecimal()`, `isnumeric()`, `isalpha()`, `isalnum()`, `isspace()`, `islower()`, `isupper()`, `istitle()`, `isidentifier()`.

**Prefix/suffix removal (3.9+):** `removeprefix()`, `removesuffix()`.
'''))

# 25 - string methods demo
cells.append(code(r'''
s = "  Hello, World!  "
print("case:", s.strip().lower(), s.strip().upper(), s.strip().title(), s.strip().swapcase())
print("strip:", repr(s.strip()), repr("xxHixx".strip("x")))

print("split:", "a,b,c".split(","), "a  b c".split())
print("join:", "-".join(["a", "b", "c"]))
print("replace:", "banana".replace("na", "NA", 1))

text = "banana"
print("find/index/count:", text.find("na"), text.index("na"), text.count("na"))
print("startswith/endswith:", "Hello".startswith("He"), "Hello".endswith("lo"))

print("predicates:", "123".isdigit(), "abc".isalpha(), "a1".isalnum(), "   ".isspace())
print("identifier:", "total_1".isidentifier(), "1total".isidentifier())
print("padding:", "7".zfill(3), "hi".center(6, "-"), "hi".rjust(5))
print("remove prefix/suffix:", "test.py".removesuffix(".py"))
'''))

# 26 - escapes / raw / f-strings
cells.append(md(r'''
## 12. Escape sequences, raw strings, and multi-line strings

An **escape sequence** starts with a backslash inside a string literal:

| Sequence | Meaning |
|---|---|
| `\n` | newline |
| `\t` | tab |
| `\r` | carriage return |
| `\\` | literal backslash |
| `\"` / `\'` | literal quote |
| `\0` | null character |
| `\uXXXX` | Unicode character by 4 hex digits |

A **raw string** `r"..."` treats backslashes as literal characters — essential for Windows paths and regular expressions: `r"C:\new\data.txt"`. (Raw strings cannot end in a single backslash.)

**Triple-quoted strings** `"""..."""` span multiple lines and are commonly used for docstrings.

**f-strings** can contain any expression (`f"{2 + 3}"`), format specs (`f"{x:.2f}"`), conversion flags `!s`/`!r`/`!a`, and even `=` for debugging (`f"{x=}"` prints `x=5`).
'''))

# 27 - escapes/raw demo
cells.append(code(r'''
print("line1\nline2")
print("col1\tcol2")
print("a quote: \"hello\" and a backslash: \\")
print("unicode: \u03c0 and \U0001F600")

path = r"C:\new\data.txt"       # raw string keeps backslashes literal
print("raw path:", path)
print("normal path:", "C:\\new\\data.txt")

doc = """This string
spans multiple
lines."""
print(doc)

name, n = "Ada", 3
print(f"{name!r} has {n+1} items; {n=}")
'''))

# 28 - conditionals
cells.append(md(r'''
## 13. Conditional statements

```python
if condition1:
    ...
elif condition2:
    ...
else:
    ...
```

`if` runs the first true branch; `elif` (else-if) chains more tests; `else` catches all remaining cases. Branches can be **nested**. Python uses indentation (4 spaces by convention) instead of braces.

**Ternary / conditional expression:** `value = a if condition else b` — a compact single-line form of `if/else`.

**Truthiness.** In a boolean context these are **falsy**: `False`, `0`, `0.0`, `0j`, `""`, `[]`, `()`, `{}`, `set()`, `None`, and objects whose `__bool__`/`__len__` return `False`/`0`. Most other objects are **truthy**. This lets you write `if name:` instead of `if name != "":`.
'''))

# 29 - conditionals demo
cells.append(code(r'''
def grade(score):
    """Return a letter grade using if/elif/else and nested logic."""
    if not 0 <= score <= 100:
        return "invalid"
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"

for value in [95, 82, 66, 120]:
    print(value, "->", grade(value))

# Ternary expression
age = 20
status = "adult" if age >= 18 else "minor"
print("status:", status)

# Truthiness
for item in [0, 1, "", "x", [], [1], None]:
    print(f"{str(item):>5} is {'truthy' if item else 'falsy'}")
'''))

# 30 - loops
cells.append(md(r'''
## 14. Loops — `for`, `while`, `range`, `enumerate`, `zip`

**`for` loop** iterates over any iterable (list, string, dict, set, file, `range`):

```python
for item in iterable:
    ...
```

Iterating a dict yields its keys; use `.items()` for key/value pairs.

**`while` loop** repeats while a condition is true — use it when the number of iterations is unknown:

```python
while condition:
    ...
```

**Helpers:**

- `range(stop)`, `range(start, stop)`, `range(start, stop, step)` — lazy integer sequences (stop is exclusive).
- `enumerate(iterable, start=0)` — yields `(index, value)` pairs.
- `zip(*iterables)` — pairs elements from several iterables, stopping at the shortest.
- Nested loops: a loop inside another loop (e.g. rows × columns).
'''))

# 31 - for/range/enumerate/zip
cells.append(code(r'''
for i in range(5):
    print(i, end=" ")
print()
print("range(2, 10, 3):", list(range(2, 10, 3)))

for index, value in enumerate(["a", "b", "c"], start=1):
    print(f"item {index}: {value}")

names = ["Ada", "Alan"]
scores = [95, 87]
for name, score in zip(names, scores):
    print(name, score)

for char in "Hi":
    print("char:", char)

person = {"name": "Ada", "role": "scientist"}
for key, value in person.items():
    print(key, "=", value)
'''))

# 32 - while + nested
cells.append(code(r'''
count = 0
while count < 3:
    print("count:", count)
    count += 1

# Nested loops: multiplication mini-table
for row in range(1, 4):
    for col in range(1, 4):
        print(f"{row * col:2d}", end=" ")
    print()
'''))

# 33 - loop control
cells.append(md(r'''
## 15. Loop control statements

- **`break`** exits the nearest loop immediately.
- **`continue`** skips the rest of the current iteration and moves to the next one.
- **`pass`** is a no-op placeholder where syntax requires a statement (empty block, stub function/class).
- **`else` on a loop** runs *after* the loop finishes **only if it was not terminated by `break`**. For a `while` loop it runs when the condition becomes false; for a `for` loop it runs when the iterable is exhausted.

The loop `else` is especially useful for "search" loops: put the found-and-stop logic in the loop body with `break`, and the "not found" logic in `else`.
'''))

# 34 - loop control demo
cells.append(code(r'''
# break: stop at first value divisible by 5
for n in range(1, 20):
    if n % 5 == 0:
        print("break at", n)
        break

# continue: print only odd numbers
for n in range(6):
    if n % 2 == 0:
        continue
    print("odd:", n)

# pass: placeholder
for n in range(3):
    pass
print("pass loop done")

# else-on-loop: detect primes
for n in range(2, 10):
    for d in range(2, n):
        if n % d == 0:
            break
    else:
        print(n, "is prime")

# while with else
k = 0
while k < 3:
    k += 1
else:
    print("while finished without break, k =", k)
'''))

# 35 - builtins survey
cells.append(md(r'''
## 16. Built-in functions survey

Python includes ~70 built-ins always available without imports. Frequently used in data work:

| Function | Purpose |
|---|---|
| `len(x)` | number of items / characters |
| `sum(iterable, start=0)` | sum of numeric items |
| `min(x)`, `max(x)` | smallest / largest item |
| `sorted(iterable, key=None, reverse=False)` | new sorted list |
| `round(x, ndigits=None)` | round to precision (bankers' rounding) |
| `abs(x)` | absolute value |
| `any(x)`, `all(x)` | truth test across an iterable |
| `map(fn, *iterables)` | apply `fn` to each item (iterator) |
| `filter(fn, iterable)` | keep items where `fn(item)` is truthy (iterator) |
| `zip(*iterables)` | combine iterables into tuples |
| `enumerate(iterable, start=0)` | add an index |
| `type(x)`, `isinstance(x, T)` | type inspection |
| `range(start, stop, step)` | integer sequence |
| `chr(n)`, `ord(c)`, `bin/hex/oct(n)` | character/integer conversions |
| `divmod(a, b)`, `pow(a, b)` | numeric helpers |
| `iter(x)`, `next(it)` | iteration protocol |

`map()` and `filter()` return **iterators**, so wrap them in `list()` to see results. `sorted()` returns a new list; `list.sort()` sorts in place. Be aware `round(2.675, 2)` gives `2.67` because of binary float representation.
'''))

# 36 - builtins demo
cells.append(code(r'''
nums = [3, 1, 4, 1, 5, 9, 2]
print("len/sum/min/max:", len(nums), sum(nums), min(nums), max(nums))
print("sorted:", sorted(nums), "reverse:", sorted(nums, reverse=True))
print("round/abs:", round(2.675, 2), abs(-5))
print("any/all:", any([0, False, 3]), all([1, True, 3]))
print("map ->", list(map(lambda v: v * v, nums)))
print("filter ->", list(filter(lambda v: v % 2 == 0, nums)))
print("enumerate ->", list(enumerate(nums)))
print("zip ->", list(zip(nums, "abcdefg")))
print("type/isinstance:", type(nums).__name__, isinstance(nums, (list, tuple)))
print("conversions:", bin(10), hex(255), oct(8), chr(65), ord("A"))
'''))

# 37 - user-defined functions
cells.append(md(r'''
## 17. User-defined functions

```python
def name(positional, keyword="default", *args, **kwargs):
    """Docstring: describe the function."""
    ...
    return value
```

- **`def`** defines a function; calling it executes the body.
- **Parameters** are names in the definition; **arguments** are values passed at the call.
- **Types of parameters:** positional (required, in order), keyword (`name=value`, order-independent), default (`param=value` if omitted), `*args` (extra positional → tuple), `**kwargs` (extra keyword → dict).
- **`return`** sends a value back and exits; without it the function returns `None`. Multiple values are returned as a tuple and can be unpacked.
- **Docstring**: a string as the first statement, accessible as `func.__doc__` and shown by `help(func)`.
- Parameter order in a definition: positional, default, `*args`, keyword-only, `**kwargs`.
- Call with `*` / `**` to unpack sequences/dicts into arguments.
'''))

# 38 - UDF demo
cells.append(code(r'''
def greet(name, greeting="Hello"):
    """Return a greeting string for the given name."""
    return f"{greeting}, {name}!"

print(greet("Ada"))
print(greet("Ada", greeting="Hi"))
print(greet(greeting="Yo", name="Ada"))   # keyword order does not matter

def stats(values):
    """Return (count, total, mean) for a sequence of numbers."""
    count = len(values)
    total = sum(values)
    return count, total, total / count if count else 0

count, total, mean = stats([1, 2, 3, 4])
print("stats:", count, total, mean)
print("docstring:", greet.__doc__)

def inspect_args(*args, **kwargs):
    """Collect arbitrary positional and keyword arguments."""
    print("args:", args)
    print("kwargs:", kwargs)

inspect_args(1, 2, 3, tag="demo", verbose=True)
print("returns None:", inspect_args())
'''))

# 39 - scope
cells.append(md(r'''
## 18. Scope — `global` and `nonlocal`

Python resolves names using the **LEGB** rule, searching these scopes in order:

1. **L**ocal — inside the current function.
2. **E**nclosing — in any enclosing (nested) function.
3. **G**lobal — at module level.
4. **B**uilt-in — names like `len`, `print`.

Reading a global variable works without declaration, but **assigning** to it inside a function creates a new local variable unless you declare `global name`. To rebind a variable in an enclosing (but non-global) scope from a nested function, use `nonlocal name`.

Avoid relying on mutable default arguments (e.g. `def f(items=[]):`), because the default is created once and shared across calls; use `None` as the default and create the list inside.
'''))

# 40 - scope demo
cells.append(code(r'''
counter = 0

def increment():
    """Modify the module-level counter using global."""
    global counter
    counter += 1

increment()
increment()
print("global counter:", counter)

def outer():
    """Demonstrate nonlocal by changing an enclosing variable."""
    value = 10
    def inner():
        nonlocal value
        value += 5
        return value
    return inner()

print("nonlocal result:", outer())
print("built-in name lookup:", len([1, 2]))
'''))

# 41 - recursion
cells.append(md(r'''
## 19. Recursion

**Recursion** is when a function calls itself. Every recursive function needs:

1. A **base case** — a simple condition that stops the recursion and returns directly.
2. A **recursive case** — the function calls itself with a smaller/simpler argument, moving toward the base case.

Classic examples:

- `factorial(n) = n * factorial(n-1)`, with `factorial(0) = 1`.
- `fibonacci(n) = fibonacci(n-1) + fibonacci(n-2)`, with `fibonacci(0)=0`, `fibonacci(1)=1`.

Every call adds a frame to the call stack, so deep recursion can hit `RecursionError` (default limit ~1000; inspect with `sys.getrecursionlimit()`). Recursion is elegant for trees and divide-and-conquer; for simple linear work an iterative loop is usually faster and safer. Naive recursive Fibonacci recomputes values exponentially — memoisation or iteration fixes that.
'''))

# 42 - recursion demo
cells.append(code(r'''
def factorial(n):
    """Return n! using recursion, with a base case at 0 and 1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n in (0, 1):
        return 1
    return n * factorial(n - 1)

print("factorials:", [factorial(n) for n in range(7)])

def fibonacci(n):
    """Return the nth Fibonacci number (naive recursive version)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print("fibonacci:", [fibonacci(n) for n in range(10)])
print("recursion limit:", sys.getrecursionlimit())
'''))

# 43 - lambda / higher-order
cells.append(md(r'''
## 20. Lambda functions and higher-order functions

A **lambda** is a small anonymous function defined with:

```python
lambda parameters: expression
```

- It can take any number of parameters but contains **one expression**, whose value is returned implicitly.
- Use lambdas for short, throwaway callbacks — especially as the `key` for `sorted()`, or with `map()` and `filter()`.
- For anything with multiple statements, loops, or a name worth documenting, prefer a normal `def`.

A **higher-order function** takes a function as an argument or returns a function. Examples in the standard library: `map()`, `filter()`, `sorted(key=...)`, `functools.reduce()`. A function can also *return* a function (a closure), which is the basis of decorators and factories such as `make_adder`.

Note: assigning a lambda to a name (`square = lambda x: x*x`) works but is generally less readable than `def square(x): return x*x`.
'''))

# 44 - lambda demo
cells.append(code(r'''
square = lambda x: x * x
print("square(7):", square(7))

pairs = [(1, "b"), (2, "a"), (0, "c")]
print("default sort:", sorted(pairs))
print("sort by second item:", sorted(pairs, key=lambda p: p[1]))

nums = [1, 2, 3, 4, 5, 6]
print("evens:", list(filter(lambda n: n % 2 == 0, nums)))
print("squares:", list(map(lambda n: n ** 2, nums)))

def make_adder(k):
    """Return a function that adds k to its argument (a closure)."""
    return lambda x: x + k

add5 = make_adder(5)
print("add5(10):", add5(10))
'''))

# 45 - Exercise 1
cells.append(md(r'''
## Try it yourself 1 — primality tester

Write a function `is_prime(n)` that returns `True` if `n` is a prime number and `False` otherwise. Use a `for` loop combined with `if`/`break`/`else` (no imports). Handle `n < 2` explicitly. Then print the primes between 1 and 20.

Think about how far you need to test divisors: checking up to `int(n ** 0.5)` is enough.
'''))

# 46 - Solution 1
cells.append(code(r'''
def is_prime(n):
    """Return True if n is prime, using trial division up to sqrt(n)."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True

print("primes 1-20:", [n for n in range(1, 21) if is_prime(n)])
'''))

# 47 - Exercise 2
cells.append(md(r'''
## Try it yourself 2 — string analysis with conditionals and a loop

Write a function `count_vowels(text)` that counts the vowels (`a, e, i, o, u`, case-insensitive) in a string using a `for` loop and a conditional. Also write a function `initials(name)` that returns the uppercase first letters of each whitespace-separated word joined by a dot (e.g. `"Ada Lovelace"` → `"A.L"`). Use string methods you learned.
'''))

# 48 - Solution 2
cells.append(code(r'''
def count_vowels(text):
    """Count vowels in text, ignoring case."""
    vowels = "aeiou"
    count = 0
    for char in text.lower():
        if char in vowels:
            count += 1
    return count

def initials(name):
    """Return dotted uppercase initials of the words in name."""
    return ".".join(word[0].upper() for word in name.split() if word)

print("vowels:", count_vowels("Data Science is fun"))
print("initials:", initials("Ada Lovelace"))
'''))

# 49 - Exercise 3
cells.append(md(r'''
## Try it yourself 3 — lambdas, built-ins, and formatting

Given `values = [12, 7, 19, 4, 25, 10, 3]`, use `filter()` with a lambda to keep only values greater than 10, then use `map()` with a lambda to square each kept value. Print the result with an f-string showing both the original and transformed lists. Finally, use `sorted()` with a `key` lambda to sort a list of `(name, score)` tuples by score descending.
'''))

# 50 - Solution 3
cells.append(code(r'''
values = [12, 7, 19, 4, 25, 10, 3]

kept = list(filter(lambda v: v > 10, values))
squared = list(map(lambda v: v ** 2, kept))
print(f"original={values} -> kept={kept} -> squared={squared}")

people = [("Ada", 95), ("Alan", 87), ("Grace", 98)]
ranked = sorted(people, key=lambda p: p[1], reverse=True)
print("ranked by score:", ranked)
'''))

# 51 - Practical DS example
cells.append(md(r'''
## 21. A practical data-science flavoured example

We combine several unit concepts in one place: a user-defined function, built-ins (`len`, `sum`, `min`, `max`), type conversion, f-string formatting, and reading a CSV row with the standard-library `csv` module. `summarize()` returns count, mean, minimum, and maximum for a list of numbers — a mini version of what pandas does with `Series.describe()`.

The CSV part reads the first two rows of the local `tips.csv` if it exists (the shared dataset folder is two levels up from this grouped notebook); otherwise it falls back to an inline sample so the cell always runs.
'''))

# 52 - practical example code
cells.append(code(r'''
def summarize(numbers):
    """Return a dict of count, mean, min and max for a numeric sequence."""
    if not numbers:
        raise ValueError("numbers must not be empty")
    return {
        "count": len(numbers),
        "mean": round(sum(numbers) / len(numbers), 2),
        "min": min(numbers),
        "max": max(numbers),
    }

readings = [18.5, 20.1, 19.3, 22.0, 17.8, 21.4]
summary = summarize(readings)
for key, value in summary.items():
    print(f"{key:>5}: {value}")

# Read a local CSV row using the csv module (guarded so the cell always runs)
csv_path = DATA / "tips.csv"
if csv_path.exists():
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source = "local tips.csv"
else:
    sample = "total_bill,tip,sex\n16.99,1.01,Female\n10.34,1.66,Male\n"
    rows = list(csv.DictReader(io.StringIO(sample)))
    source = "inline sample"

print("\nCSV source:", source)
for row in rows[:2]:
    print(f"bill={float(row['total_bill']):.2f}, tip={float(row['tip']):.2f}, sex={row['sex']}")
'''))

# 53 - Cheat sheet
cells.append(md(r'''
## Cheat sheet / key takeaways

**Data types & variables**

| Concept | Syntax |
|---|---|
| Inspect type | `type(x)`, `isinstance(x, (int, float))` |
| Literals | `42`, `3.14`, `2+3j`, `True`, `"txt"`, `b"bytes"`, `None` |
| Chained / unpack / swap | `a=b=0`, `x,y=1,2`, `a,b=b,a` |
| Starred / augmented | `first,*rest=items`, `x += 1` |

**I/O & formatting**

| Task | Syntax |
|---|---|
| Print with separators | `print(a, b, sep=", ", end="\n")` |
| f-string | `f"{name:>10} {score:.2f} {pct:.0%}"` |
| Read input | `raw = input("prompt")` → always `str` |

**Operators (lowest → highest):** `or` < `and` < `not` < comparisons < `|` < `^` < `&` < shifts < `+ -` < `* / // %` < unary < `**`. `is` = identity, `in` = membership, `==` = equality.

**Strings:** `s[i]`, `s[a:b:step]`, `s[::-1]`; case `upper/lower/title`; `strip/split/join/replace/find/index/count/startswith/endswith/isdigit`; raw `r"..."`, escapes `\n \t \\`, f-strings.

**Control flow:** `if/elif/else`, `x if c else y`; `for ... in ...`, `while`, `range`, `enumerate`, `zip`; `break`, `continue`, `pass`, loop `else`. Truthy: most values; falsy: `0`, `""`, `None`, empty containers.

**Functions:** `def f(a, b=1, *args, **kwargs): ...; return x`; docstring first; scope LEGB; `global`/`nonlocal` to rebind. Recursion needs a base case. Lambdas: `lambda x: expr`, best as `key=` callbacks and with `map`/`filter`.

**Built-ins to remember:** `len, sum, min, max, sorted, round, abs, any, all, map, filter, zip, enumerate, type, isinstance, range, divmod, pow, chr, ord`.
'''))

# 54 - Further reading
cells.append(md(r'''
## Further reading

Continue with these official sources:

- [Python 3 Tutorial (full)](https://docs.python.org/3/tutorial/index.html)
- [Python Language Reference](https://docs.python.org/3/reference/index.html)
- [Built-in types](https://docs.python.org/3/library/stdtypes.html)
- [How To Use Functional Programming Techniques](https://docs.python.org/3/howto/functional.html)
- [itertools — functions for efficient looping](https://docs.python.org/3/library/itertools.html)
- [functools — higher-order functions and callable objects](https://docs.python.org/3/library/functools.html)
- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 498 — Literal String Interpolation (f-strings)](https://peps.python.org/pep-0498/)
- [Jupyter documentation](https://docs.jupyter.org/en/latest/)
- [Python getting started guide](https://www.python.org/about/gettingstarted/)
'''))

write_notebook(OUT, cells)
