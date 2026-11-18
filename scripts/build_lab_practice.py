"""Builder for the lab-practice deliverable notebook (20 programs, Units I-V).

Run with the system Python (stdlib only):
    cd DS && python3 scripts/build_lab_practice.py
"""

from nbhelper import md, code, write_notebook

cells = []

# ======================================================================
# Front matter
# ======================================================================
cells.append(md(r"""
# Lab Practice — NWS-101/102 (Python for Data Science), Units I–V

This notebook is the **lab-practice deliverable** for the syllabus line:

> "LAB WORK: Development and execution of any 3-4 programs from each unit I, II & III, using programming concepts and implementation of any 3-4 Python libraries from each of units IV and V."

It contains **20 small programs**: four from Unit I, four from Unit II, four from Unit III, four library programs from Unit IV and four library programs from Unit V. Every program is labelled, solved in runnable code cells, and followed by a short note on what to notice. All data comes from the local `../data` folder or from synthetic arrays, so the whole notebook runs offline in a few seconds.
"""))

cells.append(md(r"""
## How to use this notebook

1. Start JupyterLab from the project root (`cd DS`), then open `notebooks/lab_practice.ipynb`. The kernel is **python3** and the environment is the project `.venv`.
2. Run the cells **top to bottom** once. The cells share a namespace, so later programs reuse variables (for example `DATA`, `iris`, `tips`) created earlier.
3. Data lives one level above `notebooks/`, hence `DATA = Path("../data")` in the first code cell. No dataset is downloaded.
4. Code cells contain explanatory `#` comments: they are part of the teaching material. The **Sample output / what to notice** cells describe the expected result; the saved notebook keeps all code outputs empty.
5. Optional steps that need a network or a database server are shown as **fenced markdown snippets only** and are never executed here.

The deep-learning programs print short training summaries; they train tiny models on synthetic data, not MNIST or any web data.
"""))

cells.append(md(r"""
## Syllabus mapping

| Syllabus clause (verbatim requirement) | Programs in this notebook | Count |
| --- | --- | --- |
| Development and execution of any **3-4 programs from unit I** | I.1 grade/temperature calculator, I.2 number & string analysis, I.3 recursion + lambda, I.4 file I/O | 4 |
| Development and execution of any **3-4 programs from unit II** | II.1 list/dict record manager, II.2 set operations, II.3 map/filter/reduce/zip pipeline, II.4 user-defined + stdlib modules | 4 |
| Development and execution of any **3-4 programs from unit III** | III.1 OOP inheritance/polymorphism, III.2 exceptions + JSON/CSV, III.3 regex cleaner, III.4 sqlite3 + SQLAlchemy ORM | 4 |
| Implementation of any **3-4 Python libraries from unit IV** | IV.1 NumPy, IV.2 pandas, IV.3 SciPy, IV.4 Matplotlib + Seaborn + SymPy + Statsmodels | 4 |
| Implementation of any **3-4 Python libraries from unit V** | V.1 scikit-learn, V.2 TensorFlow/Keras, V.3 PyTorch, V.4 NLTK + Gensim + SQLAlchemy + DuckDB | 4 |

Units I–III emphasise **programming concepts**; Units IV–V emphasise **library implementation**. Each group exceeds the minimum of three programs by providing four.
"""))

cells.append(md(r"""
## Official documentation

- Python tutorial: https://docs.python.org/3/tutorial/
- `pathlib`, `tempfile`, `re`, `csv`, `json`, `sqlite3`: https://docs.python.org/3/library/
- NumPy: https://numpy.org/doc/stable/user/absolute_beginners.html
- pandas: https://pandas.pydata.org/docs/user_guide/index.html
- SciPy: https://docs.scipy.org/doc/scipy/tutorial/index.html
- Matplotlib: https://matplotlib.org/stable/tutorials/index.html
- Seaborn: https://seaborn.pydata.org/tutorial.html
- SymPy: https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
- Statsmodels: https://www.statsmodels.org/stable/gettingstarted.html
- scikit-learn: https://scikit-learn.org/stable/getting_started.html
- TensorFlow / Keras: https://www.tensorflow.org/tutorials , https://keras.io/getting_started/
- PyTorch: https://pytorch.org/tutorials/beginner/basics/intro.html
- NLTK: https://www.nltk.org/book/
- Gensim: https://radimrehurek.com/gensim/
- SQLAlchemy 2.0: https://docs.sqlalchemy.org/en/20/orm/quickstart.html
- DuckDB: https://duckdb.org/docs/stable/python/overview.html
"""))

cells.append(code(r"""
from pathlib import Path

DATA = Path("../data")
# Every program reads data relative to the notebooks folder.
print("Data folder:", DATA.resolve())
print("Folder exists:", DATA.exists())
"""))

# ======================================================================
# UNIT I
# ======================================================================
cells.append(md(r"""
---

**UNIT I — PYTHON FUNDAMENTALS**

These four programs use only the standard library: conditionals, loops, functions, recursion, lambda expressions and simple file I/O.
"""))

# --- I.1 ---
cells.append(md(r"""
## Program I.1 — Temperature and grade calculator with conditionals
"""))
cells.append(md(r"""
**Problem statement.** Convert a list of Celsius temperatures to Fahrenheit and assign a letter grade to a list of marks, rejecting values outside 0–100.
"""))
cells.append(md(r"""
**Approach.** Write two small functions. The conversion uses the formula `F = C * 9/5 + 32`. The grade function validates the range first, then tests thresholds from highest to lowest with `if`/`elif`/`else`.
"""))
cells.append(code(r"""
# Celsius -> Fahrenheit uses the standard 9/5 scale plus an offset of 32.
def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


# Validate first, then check thresholds from the highest grade downwards.
def letter_grade(marks):
    if marks < 0 or marks > 100:
        return "invalid"
    if marks >= 90:
        return "A"
    if marks >= 80:
        return "B"
    if marks >= 70:
        return "C"
    if marks >= 50:
        return "D"
    return "F"


temperatures = [0, 20, 37, 100]
for celsius in temperatures:
    # f-strings give aligned, readable output.
    print(f"{celsius:>4} C = {celsius_to_fahrenheit(celsius):6.1f} F")

marks = [95, 82, 74, 61, 45, 120]
for value in marks:
    print(f"marks {value:>3} -> grade {letter_grade(value)}")
"""))
cells.append(md(r"""
**Sample output / what to notice.** `0 C = 32.0 F`, `37 C = 98.6 F`, `100 C = 212.0 F`. Grades come out `A, B, C, D, F` and the out-of-range `120` becomes `invalid` because the guard clause runs before any threshold.
"""))

# --- I.2 ---
cells.append(md(r"""
## Program I.2 — Number and string analysis with loops and functions
"""))
cells.append(md(r"""
**Problem statement.** Analyse one sentence (characters, words, vowels, digits, longest word), test whether several strings are palindromes, and sum the squares of the even numbers in a list.
"""))
cells.append(md(r"""
**Approach.** Use `split()` and generator expressions inside functions, reverse a cleaned string for the palindrome test, and use a `for` loop with a running accumulator for the numeric part.
"""))
cells.append(code(r"""
# Return several statistics about a sentence in one dictionary.
def analyse_text(text):
    words = text.split()
    vowels = "aeiou"
    # A generator expression counts without building an intermediate list.
    vowel_count = sum(1 for character in text.lower() if character in vowels)
    digit_count = sum(1 for character in text if character.isdigit())
    return {
        "characters": len(text),
        "words": len(words),
        "vowels": vowel_count,
        "digits": digit_count,
        "longest_word": max(words, key=len) if words else "",
    }


# Ignore spaces, punctuation and case before comparing with the reverse.
def is_palindrome(text):
    cleaned = "".join(character.lower() for character in text if character.isalnum())
    return cleaned == cleaned[::-1]


sentence = "Data science makes 2026 a fun place to learn."
print(analyse_text(sentence))

for candidate in ["level", "radar", "python", "A man a plan a canal Panama"]:
    print(f"{candidate!r:35} palindrome? {is_palindrome(candidate)}")

# A classic accumulator loop: add only the even numbers' squares.
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
total = 0
for number in numbers:
    if number % 2 == 0:
        total += number ** 2
print("sum of squares of evens:", total)
"""))
cells.append(md(r"""
**Sample output / what to notice.** The dictionary reports 45 characters, 9 words, 14 vowels, 4 digits and `science` as the longest word. `level`, `radar` and the famous Panama sentence are palindromes; `python` is not. The accumulator finishes at `4 + 16 + 36 + 64 = 120`.
"""))

# --- I.3 ---
cells.append(md(r"""
## Program I.3 — Recursion and lambda expressions
"""))
cells.append(md(r"""
**Problem statement.** Compute factorials and Fibonacci numbers recursively, then use one-line `lambda` functions to square, sort and filter a list.
"""))
cells.append(md(r"""
**Approach.** Every recursive function has a **base case** that stops the calls and a **recursive case** that calls itself on a smaller input. Lambdas are anonymous functions, useful as arguments to `map`, `filter` and `sorted`.
"""))
cells.append(code(r"""
def factorial(number):
    # Base case: 0! and 1! are both 1.
    if number < 0:
        raise ValueError("number must be non-negative")
    if number in (0, 1):
        return 1
    # Recursive case: n! = n * (n-1)!.
    return number * factorial(number - 1)


def fibonacci(number):
    # Base cases: F(0)=0 and F(1)=1.
    if number < 0:
        raise ValueError("number must be non-negative")
    if number < 2:
        return number
    # Each call branches into two smaller calls.
    return fibonacci(number - 1) + fibonacci(number - 2)


print("factorials:", [factorial(number) for number in range(7)])
print("fibonacci: ", [fibonacci(number) for number in range(10)])

# Lambdas assigned to names are still anonymous one-line functions.
square = lambda value: value ** 2
is_even = lambda value: value % 2 == 0
values = [5, 2, 8, 1, 9, 4]
print("squares:          ", list(map(square, values)))
print("sorted by distance", sorted(values, key=lambda value: abs(value - 5)))
print("filtered evens:   ", list(filter(is_even, values)))
"""))
cells.append(md(r"""
**Sample output / what to notice.** Factorials run `1, 1, 2, 6, 24, 120, 720` and Fibonacci `0, 1, 1, 2, 3, 5, 8, 13, 21, 34`. Removing either base case would cause infinite recursion (`RecursionError`). The lambda sort orders values by distance from 5, so it starts `5, 4, 2, ...`.
"""))

# --- I.4 ---
cells.append(md(r"""
## Program I.4 — Small file-based I/O program
"""))
cells.append(md(r"""
**Problem statement.** Write a short text report, read it back, count lines and words, append one more line, and verify the change.
"""))
cells.append(md(r"""
**Approach.** Use `tempfile.TemporaryDirectory()` so the program leaves no files behind. `Path.write_text` creates the file, `read_text` loads it, and `open(..., "a")` appends.
"""))
cells.append(code(r"""
import tempfile

with tempfile.TemporaryDirectory() as folder:
    report_path = Path(folder) / "lab_report.txt"
    # write_text creates the file and returns the number of characters written.
    report_path.write_text(
        "Unit I lab report\nTemperature and grade tools\nAll tests passed\n",
        encoding="utf-8",
    )
    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    print("file name: ", report_path.name)
    print("line count:", len(lines))
    print("word count:", len(text.split()))
    for index, line in enumerate(lines, start=1):
        print(f"{index}: {line}")

    # Opening in append mode adds a line without erasing the existing content.
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("Appended after review\n")
    print("after append:", len(report_path.read_text(encoding="utf-8").splitlines()), "lines")

# The temporary directory was deleted when the with-block ended.
print("temporary folder removed:", not Path(folder).exists())
"""))
cells.append(md(r"""
**Sample output / what to notice.** Three numbered lines are printed, the word count is 11, and after `open(..., "a")` the file has 4 lines. The final `True` confirms the temporary folder was cleaned up automatically.
"""))

# ======================================================================
# UNIT II
# ======================================================================
cells.append(md(r"""
---

**UNIT II — DATA STRUCTURES**

These four programs use lists, dictionaries, sets, functional built-ins, and a module written from the notebook itself plus standard-library modules.
"""))

# --- II.1 ---
cells.append(md(r"""
## Program II.1 — List/dict student record manager
"""))
cells.append(md(r"""
**Problem statement.** Keep a roster of students (each a dictionary), add a new record, compute averages, rank by average, and update an existing record.
"""))
cells.append(md(r"""
**Approach.** Represent one student as a `dict` and the whole roster as a `list` of dicts. Small functions encapsulate the add and average operations so the data layout is used consistently.
"""))
cells.append(code(r"""
# A roster is a list of dictionaries; each dictionary is one student record.
students = [
    {"name": "Asha", "marks": [88, 92, 79]},
    {"name": "Ravi", "marks": [70, 65, 81]},
]


def add_student(roster, name, marks):
    # Append a fresh record; list(marks) avoids sharing the caller's list.
    roster.append({"name": name, "marks": list(marks)})


def average_marks(record):
    # Guard against an empty marks list before dividing.
    values = record["marks"]
    return round(sum(values) / len(values), 2) if values else 0.0


add_student(students, "Meera", [95, 89, 93])
for record in students:
    print(f"{record['name']:6} marks={record['marks']} average={average_marks(record)}")

# sorted() accepts a key function, so we can rank by computed average.
ranked = sorted(students, key=average_marks, reverse=True)
print("top student:", ranked[0]["name"], average_marks(ranked[0]))

# Dictionaries are mutable: updating a nested list changes the original record.
students[1]["marks"].append(90)
print("Ravi after update:", students[1])
"""))
cells.append(md(r"""
**Sample output / what to notice.** Averages print as `86.33`, `72.0` and `92.33`; `Meera` is ranked first. The last line shows Ravi's marks list now has four entries because dicts and lists are mutable.
"""))

# --- II.2 ---
cells.append(md(r"""
## Program II.2 — Set operations on two course rosters
"""))
cells.append(md(r"""
**Problem statement.** Given the rosters of two courses, find every student, the students taking both, those taking only one course, and then check subset/disjoint relations.
"""))
cells.append(md(r"""
**Approach.** Build two `set` objects and use the operators `|` (union), `&` (intersection), `-` (difference) and `^` (symmetric difference). `sorted()` makes the printed results stable.
"""))
cells.append(code(r"""
python_course = {"Asha", "Ravi", "Meera", "Kabir"}
stats_course = {"Meera", "Kabir", "Nisha", "Dev"}

print("Python roster:", sorted(python_course))
print("Stats roster: ", sorted(stats_course))
print("union (everyone):      ", sorted(python_course | stats_course))
print("intersection (both):   ", sorted(python_course & stats_course))
print("only Python:           ", sorted(python_course - stats_course))
print("only Stats:            ", sorted(stats_course - python_course))
print("symmetric difference:  ", sorted(python_course ^ stats_course))

# The same results are available as named methods.
print("is subset?", python_course.issubset(python_course | stats_course))
print("is disjoint?", python_course.isdisjoint({"Zoe", "Yusuf"}))

# frozenset is immutable and hashable, so it can be used as a dict key.
course_key = frozenset(python_course)
print("frozenset as key:", {course_key: "Python 101"})
"""))
cells.append(md(r"""
**Sample output / what to notice.** The intersection is `['Kabir', 'Meera']` and the symmetric difference is `['Asha', 'Dev', 'Nisha', 'Ravi']` — students in exactly one course. `isdisjoint` is `True` for a roster with no shared names.
"""))

# --- II.3 ---
cells.append(md(r"""
## Program II.3 — map/filter/reduce/zip data pipeline
"""))
cells.append(md(r"""
**Problem statement.** Given parallel lists of product names, prices and quantities, build records, compute line totals, keep affordable lines, and sum the whole basket.
"""))
cells.append(md(r"""
**Approach.** `zip` aligns the parallel lists into records, `map` transforms every record, `filter` selects a subset, and `functools.reduce` folds all totals into one number. `zip(*...)` performs the inverse "unzip" step.
"""))
cells.append(code(r"""
from functools import reduce

names = ["pen", "notebook", "eraser", "marker"]
prices = [10.0, 40.0, 5.0, 25.0]
quantities = [3, 2, 5, 1]

# zip stops at the shortest input and yields (name, price, quantity) tuples.
records = list(zip(names, prices, quantities))
print("zipped records:", records)

# map applies the lambda to every record: price * quantity.
line_totals = list(map(lambda record: record[1] * record[2], records))
print("line totals:  ", line_totals)

# filter keeps only name/total pairs below the affordability threshold.
affordable = list(filter(lambda pair: pair[1] < 60, zip(names, line_totals)))
print("affordable:   ", affordable)

# reduce folds the list into a single value, starting from 0.0.
basket_total = reduce(lambda running, current: running + current, line_totals, 0.0)
print("basket total: ", basket_total)

# zip(*records) transposes rows back into columns.
columns = list(zip(*records))
print("unzipped columns:", columns)
"""))
cells.append(md(r"""
**Sample output / what to notice.** Line totals are `[30.0, 80.0, 25.0, 25.0]` and the basket sums to `160.0`. The `notebook` line is dropped by `filter` because `80 >= 60`. The unzipped result recovers the original three columns as tuples.
"""))

# --- II.4 ---
cells.append(md(r"""
## Program II.4 — A reusable user-defined module and standard-library modules
"""))
cells.append(md(r"""
**Problem statement.** Create a small Python module from inside the notebook, import and use it, then demonstrate a few standard-library modules.
"""))
cells.append(md(r"""
**Approach.** Write the module's source with `Path.write_text` into a temporary directory, prepend that directory to `sys.path`, and load the file with `importlib.import_module`. The second half uses `math`, `statistics` and `datetime` from the standard library. The temporary directory is removed automatically.
"""))
cells.append(code(r"""
import importlib
import sys
import tempfile

# This string is the full source of our own .py module.
module_source = '''
# A tiny user-defined module created from the notebook.

def double(value):
    return value * 2

def describe(value):
    return "even" if value % 2 == 0 else "odd"
'''

# Drop a stale copy first in case this cell is run more than once.
sys.modules.pop("lab_utils", None)
with tempfile.TemporaryDirectory() as folder:
    module_path = Path(folder) / "lab_utils.py"
    module_path.write_text(module_source, encoding="utf-8")
    # Python imports from the directories listed in sys.path.
    sys.path.insert(0, folder)
    try:
        lab_utils = importlib.import_module("lab_utils")
        print("module file:", module_path.name)
        print("double(7): ", lab_utils.double(7))
        print("describe(7):", lab_utils.describe(7))
        print("double(6): ", lab_utils.double(6))
    finally:
        sys.path.remove(folder)

# --- standard-library modules, no installation needed ---
import math
import statistics
import datetime

print("math.sqrt(49):    ", math.sqrt(49))
print("statistics.mean:  ", statistics.mean([2, 4, 6, 8]))
print("today's date:     ", datetime.date.today().isoformat())
print("temporary folder removed:", not Path(folder).exists())
"""))
cells.append(md(r"""
**Sample output / what to notice.** The imported module returns `14`, `odd` and `12`, proving the file written at runtime was genuinely imported. The stdlib section prints `7.0`, `5` and today's ISO date. The final `True` shows the module file did not persist.
"""))

# ======================================================================
# UNIT III
# ======================================================================
cells.append(md(r"""
---

**UNIT III — OOP, FILES, REGEX AND SQL**

These four programs cover classes with inheritance and polymorphism, exception handling around JSON/CSV files, regular expressions, and SQL through both `sqlite3` and the SQLAlchemy 2.0 ORM.
"""))

# --- III.1 ---
cells.append(md(r"""
## Program III.1 — OOP class hierarchy with inheritance and polymorphism
"""))
cells.append(md(r"""
**Problem statement.** Build a `Shape` hierarchy (`Circle`, `Rectangle`) and a `BankAccount` hierarchy (`BankAccount`, `SavingsAccount`) that show inheritance, method overriding and polymorphism.
"""))
cells.append(md(r"""
**Approach.** The base `Shape` declares `area()` and a `describe()` method that calls it. Subclasses override `area()`, so the same `describe()` call produces different results. `SavingsAccount` inherits `deposit` from `BankAccount` but overrides `withdraw` to enforce a minimum balance, calling `super().__init__` to reuse the parent constructor.
"""))
cells.append(code(r"""
import math


class Shape:
    # Base class: concrete subclasses must implement area().
    def area(self):
        raise NotImplementedError("subclasses must implement area()")

    # Polymorphism: self.area() resolves to the subclass implementation.
    def describe(self):
        return f"{type(self).__name__} area = {self.area():.2f}"


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


shapes = [Circle(2), Rectangle(3, 4), Circle(1)]
for shape in shapes:
    # One call, three different behaviours -> polymorphism.
    print(shape.describe())
print("total area:", round(sum(shape.area() for shape in shapes), 2))


class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount

    def summary(self):
        return f"{self.owner}: balance {self.balance:.2f}"


class SavingsAccount(BankAccount):
    # Inheritance: reuse the parent constructor, then add a new attribute.
    def __init__(self, owner, balance=0.0, minimum_balance=100.0):
        super().__init__(owner, balance)
        self.minimum_balance = minimum_balance

    # Overriding: a stricter rule than the parent's withdraw().
    def withdraw(self, amount):
        if self.balance - amount < self.minimum_balance:
            raise ValueError("withdrawal would break the minimum balance")
        self.balance -= amount


account = BankAccount("Asha", 500)
account.deposit(50)
account.withdraw(120)
print(account.summary())

savings = SavingsAccount("Ravi", 500)
try:
    savings.withdraw(450)
except ValueError as error:
    print("savings withdrawal blocked:", error)
savings.withdraw(200)
print(savings.summary())
"""))
cells.append(md(r"""
**Sample output / what to notice.** Areas print as `Circle area = 12.57`, `Rectangle area = 12.00`, `Circle area = 3.14`, totalling `27.71`. The ordinary account ends at `430.00`. The savings withdrawal of 450 is blocked by the override; after a valid withdrawal of 200 the balance is `300.00`, never below the 100 minimum.
"""))

# --- III.2 ---
cells.append(md(r"""
## Program III.2 — Exception handling with JSON and CSV files
"""))
cells.append(md(r"""
**Problem statement.** Read a CSV of marks where one row is malformed, skip only that row, then parse a JSON profile and show that `finally` always runs.
"""))
cells.append(md(r"""
**Approach.** Use `csv.DictReader` and `json.loads` on files written into a temporary directory. `try`/`except` isolates bad records so one error does not abort the whole file, and `finally` holds cleanup-style code that runs either way.
"""))
cells.append(code(r"""
import csv
import json
import tempfile

with tempfile.TemporaryDirectory() as folder:
    # A deliberately malformed row: "not-a-number" cannot become a float.
    csv_path = Path(folder) / "marks.csv"
    csv_path.write_text(
        "name,mark\nAsha,88\nRavi,not-a-number\nMeera,92\n",
        encoding="utf-8",
    )

    total = 0.0
    count = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                # Convert the mark; a bad value must not stop the loop.
                mark = float(row["mark"])
            except (ValueError, TypeError) as error:
                print(f"skipping {row.get('name')}: {error}")
                continue
            total += mark
            count += 1
    print("valid rows:", count, "average:", round(total / count, 2))

    # JSON is parsed from a string with json.loads.
    json_path = Path(folder) / "profile.json"
    json_path.write_text('{"name": "Asha", "skills": ["Python", "SQL"]}', encoding="utf-8")
    try:
        profile = json.loads(json_path.read_text(encoding="utf-8"))
        print("profile:", profile, "first skill:", profile["skills"][0])
    except json.JSONDecodeError as error:
        print("invalid JSON:", error)
    finally:
        # This runs on success and on failure.
        print("JSON processing attempted (finally always runs).")
"""))
cells.append(md(r"""
**Sample output / what to notice.** The Ravi row is skipped with a `ValueError`, the two valid rows average `90.0`, and the JSON profile's first skill is `Python`. The `finally` message always appears — useful for closing files or releasing resources.
"""))

# --- III.3 ---
cells.append(md(r"""
## Program III.3 — Regex-based text cleaner and validator
"""))
cells.append(md(r"""
**Problem statement.** Validate email addresses and roll numbers with regular expressions, collapse messy whitespace, and extract years and words from a sentence.
"""))
cells.append(md(r"""
**Approach.** Compile patterns once with `re.compile`, use `fullmatch` for whole-string validation, `re.sub` to replace repeated whitespace, and `re.findall` to extract all matches.
"""))
cells.append(code(r"""
import re

# An email: local part, one @, a dotted domain, and a 2+ letter extension.
email_pattern = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
for candidate in ["asha@example.com", "bad@@mail", "ravi.kumar@college.edu", "no-at-sign"]:
    # fullmatch requires the entire string to satisfy the pattern.
    print(f"{candidate:28} valid email? {bool(email_pattern.fullmatch(candidate))}")

messy = "  Hello,   WORLD!!  Python 3.12  is  fun in 2026...  "
# \s+ matches one or more spaces/tabs; replacing with one space tidies the text.
cleaned = re.sub(r"\s+", " ", messy).strip()
print("cleaned:", cleaned)

# \\b is a word boundary; findall returns every matching substring.
print("years:", re.findall(r"\b20\d{2}\b", messy))
print("words:", re.findall(r"[A-Za-z]+", cleaned))

# Roll numbers look like NWS-101/042.
roll_pattern = re.compile(r"^[A-Z]{3}-\d{3}/\d{3}$")
for roll in ["NWS-101/042", "nws-101/42", "CSE-201/007"]:
    print(f"{roll:14} valid roll? {bool(roll_pattern.fullmatch(roll))}")
"""))
cells.append(md(r"""
**Sample output / what to notice.** The two well-formed emails and both uppercase roll numbers validate; `bad@@mail`, `no-at-sign` and `nws-101/42` are rejected. `re.sub` turns the jagged spacing into single spaces, the extracted year is `2026`, and the word list is `Hello, WORLD, Python, is, fun, in`.
"""))

# --- III.4 ---
cells.append(md(r"""
## Program III.4 — SQL with sqlite3 in memory and SQLAlchemy 2.0 ORM
"""))
cells.append(md(r"""
**Problem statement.** Create a students table, insert and query rows with plain `sqlite3`, then express the same idea with the SQLAlchemy 2.0 declarative ORM. No external database server is used.
"""))
cells.append(md(r"""
**Approach.** `sqlite3.connect(":memory:")` gives a throwaway database. For the ORM part, define a `DeclarativeBase`, map a class to a table with `Mapped`/`mapped_column`, create the schema with `metadata.create_all`, and query through a `Session` using `select`.
"""))
cells.append(code(r"""
import sqlite3

# --- plain sqlite3, entirely in memory ---
connection = sqlite3.connect(":memory:")
connection.execute(
    "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, marks INTEGER)"
)
# Parameterised inserts avoid manual string formatting of values.
connection.executemany(
    "INSERT INTO students (name, marks) VALUES (?, ?)",
    [("Asha", 88), ("Ravi", 74), ("Meera", 95)],
)
rows = connection.execute(
    "SELECT name, marks FROM students WHERE marks >= ? ORDER BY marks DESC", (80,)
).fetchall()
print("sqlite3 rows above 80:", rows)
print("sqlite3 row count:", connection.execute("SELECT COUNT(*) FROM students").fetchone()[0])
connection.close()

# --- SQLAlchemy 2.0 ORM, also in memory ---
import sqlalchemy
from sqlalchemy import create_engine, select, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

print("SQLAlchemy version:", sqlalchemy.__version__)


class Base(DeclarativeBase):
    pass


class Student(Base):
    __tablename__ = "students_orm"
    # The typed 2.0 style declares columns through annotations.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    marks: Mapped[int] = mapped_column(Integer)


engine = create_engine("sqlite://", echo=False)  # in-memory SQLite
Base.metadata.create_all(engine)                  # create the table from the class
with Session(engine) as session:
    session.add_all([
        Student(name="Asha", marks=88),
        Student(name="Ravi", marks=74),
        Student(name="Meera", marks=95),
    ])
    session.commit()
    statement = select(Student).where(Student.marks >= 80).order_by(Student.marks.desc())
    for student in session.scalars(statement):
        print(f"ORM: {student.name} scored {student.marks}")
"""))
cells.append(md(r"""
**Sample output / what to notice.** Both approaches return `[('Meera', 95), ('Asha', 88)]`. The ORM version produces the same result with Python objects instead of raw SQL strings, and `create_engine("sqlite://")` keeps everything in memory — no files, no server.
"""))
cells.append(md(r"""
**Optional — not executed (needs a network/server).** To point SQLAlchemy at a real database instead of in-memory SQLite, you would need a running server and credentials. That step is intentionally shown only as a fenced snippet:

```python
# from sqlalchemy import create_engine
# engine = create_engine("postgresql+psycopg2://user:password@localhost:5432/mydb")
# engine = create_engine("mysql+pymysql://user:password@localhost:3306/mydb")
```
"""))

# ======================================================================
# UNIT IV
# ======================================================================
cells.append(md(r"""
---

**UNIT IV — PYTHON LIBRARIES (4 libraries)**

Each program below showcases a different library: NumPy, pandas, SciPy, and a combined Matplotlib + Seaborn + SymPy + Statsmodels demo.
"""))

# --- IV.1 ---
cells.append(md(r"""
## Program IV.1 — NumPy vectorised computation
"""))
cells.append(md(r"""
**Problem statement.** Load the numeric columns of the local `iris.csv` and compute column statistics, standardised values and distances from the first flower without Python loops.
"""))
cells.append(md(r"""
**Approach.** `np.genfromtxt` reads the CSV into a 2-D array. Reductions such as `mean(axis=0)` collapse columns, broadcasting subtracts the mean vector from every row, `np.where` guards against zero standard deviations, and `np.linalg.norm` computes all distances at once.
"""))
cells.append(code(r"""
import numpy as np

# Read only the four numeric columns; skip the header and the species column.
features = np.genfromtxt(DATA / "iris.csv", delimiter=",", skip_header=1,
                         usecols=(0, 1, 2, 3))
print("shape:", features.shape, "dtype:", features.dtype)

# Reductions with axis=0 produce one value per column.
column_means = features.mean(axis=0)
column_stds = features.std(axis=0)
print("means:", np.round(column_means, 3))
print("stds: ", np.round(column_stds, 3))

# Broadcasting: a length-4 vector is subtracted from every one of the 150 rows.
centred = features - column_means
safe_stds = np.where(column_stds == 0, 1, column_stds)
standardised = centred / safe_stds
print("standardised column means:", np.round(standardised.mean(axis=0), 6))

# Distance of every flower from flower 0, computed across axis 1.
distances = np.linalg.norm(features - features[0], axis=1)
print("distances from flower 0 (first 5):", np.round(distances[:5], 3))

# The same sum with an explicit loop, shown for comparison.
loop_total = 0.0
for row in features:
    loop_total += row[0] + row[1] + row[2] + row[3]
print("loop total:", round(loop_total, 2), "vector total:", round(features.sum(), 2))
"""))
cells.append(md(r"""
**Sample output / what to notice.** The array is `(150, 4)` of floats. After standardisation every column mean is essentially `0` (to six decimals), and the first distance is exactly `0.0` because it measures flower 0 against itself. The loop total and `features.sum()` match, but the vectorised form needs no Python-level iteration.
"""))

# --- IV.2 ---
cells.append(md(r"""
## Program IV.2 — pandas EDA on local iris and tips CSVs
"""))
cells.append(md(r"""
**Problem statement.** Explore two local datasets: summarise iris measurements by species and tips by day and smoker status, then derive and filter a tip-percentage column.
"""))
cells.append(md(r"""
**Approach.** `read_csv` loads the files into DataFrames. `value_counts`, `describe`, `groupby().agg()` and `pivot_table` provide the exploratory summaries. A derived column is created by vectorised arithmetic and filtered with `.loc`.
"""))
cells.append(code(r"""
import pandas as pd

iris = pd.read_csv(DATA / "iris.csv")
tips = pd.read_csv(DATA / "tips.csv")

print("iris shape:", iris.shape)
print("iris species counts:\n", iris["species"].value_counts())
print("iris numeric summary:\n", iris.describe().round(2))

# Named aggregation gives readable column names in the result.
species_summary = iris.groupby("species").agg(
    mean_sepal=("sepal_length", "mean"),
    mean_petal=("petal_length", "mean"),
    count=("species", "size"),
).round(2)
print("species summary:\n", species_summary)

# pivot_table reshapes mean tips into a day-by-smoker grid.
tip_table = tips.pivot_table(values="tip", index="day", columns="smoker",
                                 aggfunc="mean").round(2)
print("mean tip by day/smoker:\n", tip_table)

# Derive a percentage column and inspect the unusually generous tips.
tips["tip_percent"] = 100 * tips["tip"] / tips["total_bill"]
print("largest tip percent:", round(tips["tip_percent"].max(), 2))
print("high-tip rows:\n",
      tips.loc[tips["tip_percent"] > 25, ["total_bill", "tip", "day"]].head())
"""))
cells.append(md(r"""
**Sample output / what to notice.** Iris has 150 rows split evenly across three species, with setosa showing a much smaller mean petal length. The tips pivot table shows separate averages for smokers and non-smokers by day. `tip_percent` is a vectorised column, and the filtered view shows only bills where the tip exceeded 25%.
"""))

# --- IV.3 ---
cells.append(md(r"""
## Program IV.3 — SciPy statistical tests
"""))
cells.append(md(r"""
**Problem statement.** Test whether petal lengths differ between iris species and measure the correlation between sepal and petal length using `scipy.stats`.
"""))
cells.append(md(r"""
**Approach.** Extract one group per species and apply Welch's two-sample t-test (no equal-variance assumption), a one-way ANOVA across all three groups, and Pearson's correlation on two numeric columns.
"""))
cells.append(code(r"""
from scipy import stats

setosa = iris.loc[iris["species"] == "setosa", "petal_length"]
versicolor = iris.loc[iris["species"] == "versicolor", "petal_length"]
virginica = iris.loc[iris["species"] == "virginica", "petal_length"]

# Welch's t-test compares two independent groups with unequal variances.
result = stats.ttest_ind(setosa, versicolor, equal_var=False)
print("Welch t-test setosa vs versicolor")
print("  statistic:", round(result.statistic, 3), " p-value:", result.pvalue)

# ANOVA asks whether any of the three group means differ.
f_statistic, anova_p = stats.f_oneway(setosa, versicolor, virginica)
print(f"ANOVA across species: F={f_statistic:.2f}, p={anova_p:.3e}")

# Pearson correlation between two measurements of the same flowers.
correlation = stats.pearsonr(iris["sepal_length"], iris["petal_length"])
print(f"Pearson r={correlation.statistic:.3f}, p={correlation.pvalue:.3e}")
print("t-test p-value below 0.05?", result.pvalue < 0.05)
"""))
cells.append(md(r"""
**Sample output / what to notice.** The Welch statistic is strongly negative with a p-value far below `0.05`, so setosa and versicolor petal lengths genuinely differ. ANOVA also returns a tiny p-value, and Pearson's r is about `0.87`, showing a strong positive relationship between sepal and petal length.
"""))

# --- IV.4 ---
cells.append(md(r"""
## Program IV.4 — Matplotlib + Seaborn + SymPy + Statsmodels regression demo
"""))
cells.append(md(r"""
**Problem statement.** Fit a straight line to the local tips data with Statsmodels, show it with a Seaborn regression plot inside a Matplotlib axes, and differentiate a symbolic expression with SymPy.
"""))
cells.append(md(r"""
**Approach.** `sm.add_constant` adds an intercept column for `sm.OLS`. `sns.regplot` draws the scatter and fitted line on the Matplotlib axes. SymPy differentiates `x**2 + 3*x + 1` symbolically and `lambdify` turns the derivative into a NumPy-callable function.
"""))
cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import statsmodels.api as sm

# --- Statsmodels: ordinary least squares on the local tips data ---
# add_constant prepends a column of ones so the model estimates an intercept.
predictor = sm.add_constant(tips[["total_bill"]])
fitted = sm.OLS(tips["tip"], predictor).fit()
print("OLS intercept/slope:", np.round(fitted.params.values, 4))
print("OLS R-squared:", round(fitted.rsquared, 4))

# --- SymPy: symbolic differentiation, then numerical evaluation ---
symbol = sp.Symbol("x")
expression = symbol ** 2 + 3 * symbol + 1
derivative = sp.diff(expression, symbol)
print("expression:", expression, "derivative:", derivative)
print("derivative at x=2:", derivative.subs(symbol, 2))
evaluate = sp.lambdify(symbol, derivative, "numpy")
print("derivative on [1, 2, 3]:", evaluate(np.array([1, 2, 3])))

# --- Matplotlib + Seaborn: scatter with a fitted regression line ---
sns.set_theme(style="whitegrid")
figure, axes = plt.subplots(figsize=(6, 4))
sns.regplot(
    data=tips, x="total_bill", y="tip", ax=axes,
    scatter_kws={"alpha": 0.5, "s": 20}, line_kws={"color": "crimson"},
)
axes.set_title("Tip vs total bill (Seaborn regression line)")
figure.tight_layout()
plt.show()
"""))
cells.append(md(r"""
**Sample output / what to notice.** OLS estimates an intercept near `0.92` and a slope near `0.105`, with a low R-squared (tips vary a lot). SymPy reports the derivative `2*x + 3`, giving `7` at `x=2` and `[5, 7, 9]` on the sample array. The plot shows the crimson fitted line through the scatter.
"""))

# ======================================================================
# UNIT V
# ======================================================================
cells.append(md(r"""
---

**UNIT V — PYTHON LIBRARIES (4 library programs)**

These four programs implement scikit-learn, TensorFlow/Keras, PyTorch, and a combination of NLTK, Gensim, SQLAlchemy and DuckDB. All models are tiny and use local or synthetic data.
"""))

# --- V.1 ---
cells.append(md(r"""
## Program V.1 — scikit-learn classification pipeline on iris.csv
"""))
cells.append(md(r"""
**Problem statement.** Train, scale and evaluate a logistic-regression classifier that predicts iris species from the four measurements.
"""))
cells.append(md(r"""
**Approach.** Use a stratified train/test split so class proportions are preserved, wrap `StandardScaler` and `LogisticRegression` in a `Pipeline` (so scaling is fitted on training data only), and report accuracy, a confusion matrix and a classification report.
"""))
cells.append(code(r"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris_ml = pd.read_csv(DATA / "iris.csv")
feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
features_ml = iris_ml[feature_columns]
labels_ml = iris_ml["species"]

# stratify keeps the three species in the same proportions in both splits.
features_train, features_test, labels_train, labels_test = train_test_split(
    features_ml, labels_ml, test_size=0.25, random_state=0, stratify=labels_ml
)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200, random_state=0)),
])
pipeline.fit(features_train, labels_train)
predictions = pipeline.predict(features_test)

print("test accuracy:", round(accuracy_score(labels_test, predictions), 3))
print("confusion matrix (rows=true, columns=predicted):\n",
      confusion_matrix(labels_test, predictions))
print("classification report:\n", classification_report(labels_test, predictions, digits=3))
"""))
cells.append(md(r"""
**Sample output / what to notice.** Accuracy is typically about `0.87–0.95` on the 38 held-out flowers. The confusion matrix is nearly diagonal; most errors come from confusing versicolor with virginica, which have overlapping petal measurements. Wrapping scaling in the pipeline prevents data leakage from the test set.
"""))

# --- V.2 ---
cells.append(md(r"""
## Program V.2 — TensorFlow/Keras small neural network
"""))
cells.append(md(r"""
**Problem statement.** Train a tiny neural network to separate two synthetic classes defined by `x0 + x1 > 0`, and report the final loss and accuracy.
"""))
cells.append(md(r"""
**Approach.** Generate reproducible points with NumPy, build a `Sequential` model with one hidden layer, compile it with `binary_crossentropy`, and train for three quiet epochs. This uses only in-memory arrays — no MNIST or downloads.
"""))
cells.append(code(r"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # keep TensorFlow's startup logs quiet

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)

# Synthetic binary data: label 1 when the two features sum to a positive value.
rng = np.random.default_rng(0)
features_nn = rng.normal(size=(200, 2))
labels_nn = (features_nn[:, 0] + features_nn[:, 1] > 0).astype(int)

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(features_nn, labels_nn, epochs=3, batch_size=16, verbose=0)
loss, accuracy = model.evaluate(features_nn, labels_nn, verbose=0)

print("final loss:", round(loss, 4), "accuracy:", round(accuracy, 3))
print("loss per epoch:", [round(value, 4) for value in history.history["loss"]])
"""))
cells.append(md(r"""
**Sample output / what to notice.** The printed TensorFlow version confirms the import worked. Loss falls across the three epochs and accuracy is well above chance (the decision boundary is a straight line, which this network can represent easily). `verbose=0` keeps the output short and the run takes only a couple of seconds.
"""))

# --- V.3 ---
cells.append(md(r"""
## Program V.3 — PyTorch small regression model
"""))
cells.append(md(r"""
**Problem statement.** Fit a one-neuron linear model to synthetic data drawn from `y = 3x + 1` plus noise, using PyTorch tensors, autograd and SGD.
"""))
cells.append(md(r"""
**Approach.** Create tensors with `torch.linspace`, define `torch.nn.Linear(1, 1)`, and run a manual training loop: zero gradients, predict, compute MSE, call `backward()`, then `optimizer.step()`.
"""))
cells.append(code(r"""
import torch

torch.manual_seed(0)  # reproducible noise and initial weights

# Synthetic regression target: a line with a small amount of noise.
x_values = torch.linspace(-1, 1, 40).unsqueeze(1)  # shape (40, 1)
targets = 3 * x_values + 1 + 0.1 * torch.randn_like(x_values)

model_torch = torch.nn.Linear(1, 1)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model_torch.parameters(), lr=0.1)

for step in range(60):
    # Gradients accumulate by default, so clear them before each step.
    optimizer.zero_grad()
    predictions_torch = model_torch(x_values)
    loss = loss_function(predictions_torch, targets)
    loss.backward()   # autograd computes d(loss)/d(parameters)
    optimizer.step()  # SGD updates weight and bias
    if step % 20 == 0:
        print(f"step {step:2d} loss {loss.item():.4f}")

print("final loss:", round(loss.item(), 4))
print("learned weight/bias:", round(model_torch.weight.item(), 3),
      round(model_torch.bias.item(), 3))
print("expected weight/bias: 3.0 and 1.0")
"""))
cells.append(md(r"""
**Sample output / what to notice.** The loss prints roughly every 20 steps and drops steadily. The learned weight approaches `3.0` and the bias approaches `1.0`, recovering the rule that generated the data. The manual loop shows what `model.fit` hides in the Keras example.
"""))

# --- V.4 ---
cells.append(md(r"""
## Program V.4 — NLTK + Gensim + SQLAlchemy + DuckDB text and data demo
"""))
cells.append(md(r"""
**Problem statement.** Tokenise three short documents with NLTK, build a bag-of-words/TF-IDF representation with Gensim, store the documents with the SQLAlchemy ORM, and aggregate word counts with DuckDB.
"""))
cells.append(md(r"""
**Approach.** `wordpunct_tokenize` needs no downloaded corpora, so it stays offline. Gensim's `Dictionary`, `doc2bow` and `TfidfModel` convert tokens to weighted vectors. SQLAlchemy stores each document in an in-memory SQLite table, and DuckDB runs SQL directly over a registered pandas DataFrame.
"""))
cells.append(code(r"""
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sqlalchemy import create_engine, select, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import duckdb
import pandas as pd

documents = [
    "Python is great for data science and Python is fun.",
    "SQL and Python work well together in data analysis.",
    "Gensim builds topics from text and DuckDB queries data.",
]

# wordpunct_tokenize splits on punctuation without needing NLTK data downloads.
tokenized = [wordpunct_tokenize(document.lower()) for document in documents]
frequency = FreqDist(token for tokens in tokenized for token in tokens)
print("most common words:", frequency.most_common(6))

# Gensim: map tokens to integer ids, then weight them with TF-IDF.
dictionary = Dictionary(tokenized)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
tfidf = TfidfModel(corpus)
print("vocabulary size:", len(dictionary))
print("first BOW:     ", corpus[0][:5])
print("first TF-IDF:  ", tfidf[corpus[0]][:5])

# SQLAlchemy 2.0 ORM: store documents in an in-memory SQLite table.
class DocumentBase(DeclarativeBase):
    pass


class Document(DocumentBase):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[int] = mapped_column()


engine_text = create_engine("sqlite://")
DocumentBase.metadata.create_all(engine_text)
with Session(engine_text) as session:
    session.add_all([
        Document(text=text, words=len(tokens))
        for text, tokens in zip(documents, tokenized)
    ])
    session.commit()
    for item in session.scalars(select(Document).order_by(Document.words.desc())):
        print(f"stored doc {item.id}: {item.words} words")

# DuckDB: run SQL against a registered pandas DataFrame, with no server.
word_frame = pd.DataFrame({
    "document": range(len(documents)),
    "words": [len(tokens) for tokens in tokenized],
})
connection_duck = duckdb.connect()
connection_duck.register("word_frame", word_frame)
result = connection_duck.execute(
    "SELECT SUM(words) AS total, AVG(words) AS average FROM word_frame"
).fetchall()
connection_duck.close()
print("DuckDB total/average words:", result)
"""))
cells.append(md(r"""
**Sample output / what to notice.** `Python` and `data` are the most frequent tokens. Gensim reports a vocabulary of roughly 20 unique words and shows integer ids paired with TF-IDF weights. The ORM prints three stored documents sorted by word count, and DuckDB returns the total and average word counts as one row — all without a server or internet access.
"""))
cells.append(md(r"""
**Optional — not executed (needs internet).** If you later want NLTK corpora such as `punkt` or `stopwords`, download them outside this offline lab:

```python
# import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
```
"""))

# ======================================================================
# Closing checklist
# ======================================================================
cells.append(md(r"""
## Checklist

Every program is mapped to the exact lab-work syllabus line it satisfies.

| Program | Title | Syllabus line satisfied |
| --- | --- | --- |
| I.1 | Temperature and grade calculator | "Development and execution of any 3-4 programs from unit I" |
| I.2 | Number and string analysis | "Development and execution of any 3-4 programs from unit I" |
| I.3 | Recursion and lambda | "Development and execution of any 3-4 programs from unit I" |
| I.4 | File-based I/O | "Development and execution of any 3-4 programs from unit I" |
| II.1 | List/dict student records | "Development and execution of any 3-4 programs from unit II" |
| II.2 | Set operations on rosters | "Development and execution of any 3-4 programs from unit II" |
| II.3 | map/filter/reduce/zip pipeline | "Development and execution of any 3-4 programs from unit II" |
| II.4 | User-defined + stdlib modules | "Development and execution of any 3-4 programs from unit II" |
| III.1 | OOP inheritance/polymorphism | "Development and execution of any 3-4 programs from unit III" |
| III.2 | Exceptions + JSON/CSV files | "Development and execution of any 3-4 programs from unit III" |
| III.3 | Regex cleaner/validator | "Development and execution of any 3-4 programs from unit III" |
| III.4 | sqlite3 + SQLAlchemy ORM | "Development and execution of any 3-4 programs from unit III" |
| IV.1 | NumPy vectorised computation | "implementation of any 3-4 Python libraries from unit IV" |
| IV.2 | pandas EDA on local CSVs | "implementation of any 3-4 Python libraries from unit IV" |
| IV.3 | SciPy statistical tests | "implementation of any 3-4 Python libraries from unit IV" |
| IV.4 | Matplotlib + Seaborn + SymPy + Statsmodels | "implementation of any 3-4 Python libraries from unit IV" |
| V.1 | scikit-learn classification pipeline | "implementation of any 3-4 Python libraries from unit V" |
| V.2 | TensorFlow/Keras small model | "implementation of any 3-4 Python libraries from unit V" |
| V.3 | PyTorch small model | "implementation of any 3-4 Python libraries from unit V" |
| V.4 | NLTK + Gensim + SQLAlchemy + DuckDB | "implementation of any 3-4 Python libraries from unit V" |

**Total: 20 programs — 4 per unit group**, satisfying both the "3-4 programs from each unit I, II & III" clause and the "3-4 Python libraries from each of units IV and V" clause of the lab-work requirement.
"""))

write_notebook("notebooks/lab_practice.ipynb", cells)
