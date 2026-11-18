"""Builder for unit3_oops_files_regex_sql.ipynb (stdlib only, system python3)."""

from nbhelper import md, code, write_notebook

cells = []

# ---------------------------------------------------------------------------
# 1. Title, intro, learning objectives
# ---------------------------------------------------------------------------
cells.append(md(r'''
# Unit III — Fundamentals of Python Language 3: OOP, Files, Regex & SQL

This reference notebook covers the third unit of the syllabus: object-oriented programming and the
practical tooling that surrounds it. We build up from classes and objects through inheritance,
polymorphism, abstraction and encapsulation, then move to exception handling, context managers,
file and CSV/JSON I/O, regular expressions, and relational databases with SQLite, SQLAlchemy 2.0,
and templates for MySQL and PostgreSQL. Every code cell is self-contained and runnable offline;
database cells use SQLite in memory so no server is required. Read it top to bottom once, then
return to individual sections as a cheat sheet.

## Learning objectives
- Define classes, create objects, and explain the role of `self` and instance methods.
- Distinguish class attributes from instance attributes and use the `__init__` constructor.
- Implement `__repr__` / `__str__`, `@classmethod`, `@staticmethod`, and `@property` (with setters).
- Apply single and multiple inheritance, `super()`, and method overriding.
- Explain polymorphism through duck typing and operator overloading with dunder methods.
- Use abstraction with `abc.ABC` and `@abstractmethod`.
- Apply encapsulation via private conventions and name mangling.
- Use `@dataclass`, `field()`, and `__slots__` to reduce boilerplate and memory use.
- Handle errors with `try/except/else/finally`, `raise`, custom exception classes, and built-in exceptions.
- Write context managers with `with`, `__enter__`/`__exit__`, and `contextlib.contextmanager`.
- Work with files: modes, `read`/`readline`/`readlines`, `write`/`writelines`, `seek`/`tell`, and `pathlib.Path`.
- Read and write CSV with the `csv` module and JSON with the `json` module.
- Build regex patterns with character classes, quantifiers, anchors, groups, raw strings, and flags; use
  `search`, `match`, `fullmatch`, `findall`, `finditer`, `sub`, and `split`; recognise common email/phone/date patterns.
- Explain relational concepts and write `CREATE`, `INSERT`, `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`,
  `JOIN`, `UPDATE`, and `DELETE` in SQL.
- Use `sqlite3` in memory, SQLAlchemy 2.0 Core and ORM, and read MySQL/PostgreSQL connection templates.
'''))

# ---------------------------------------------------------------------------
# 2. Official documentation & references
# ---------------------------------------------------------------------------
cells.append(md(r'''
## Official documentation & references

- [Python 3 tutorial — Classes](https://docs.python.org/3/tutorial/classes.html)
- [Python 3 — Built-in exceptions](https://docs.python.org/3/library/exceptions.html)
- [Python 3 tutorial — Input and Output (files)](https://docs.python.org/3/tutorial/inputoutput.html)
- [Python 3 — `re` regular expressions](https://docs.python.org/3/library/re.html)
- [Python 3 — `sqlite3`](https://docs.python.org/3/library/sqlite3.html)
- [SQLAlchemy 2.0 tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Python 3 — `dataclasses`](https://docs.python.org/3/library/dataclasses.html)
- [Python 3 — `abc` abstract base classes](https://docs.python.org/3/library/abc.html)
- [Python 3 — `pathlib`](https://docs.python.org/3/library/pathlib.html)
- [Python 3 — `csv`](https://docs.python.org/3/library/csv.html)
- [Python 3 — `json`](https://docs.python.org/3/library/json.html)
- [Python 3 — `contextlib`](https://docs.python.org/3/library/contextlib.html)
- [Python 3 — `io` (StringIO)](https://docs.python.org/3/library/io.html)
- [SQLAlchemy 2.0 — ORM quick start](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
'''))

# ---------------------------------------------------------------------------
# 3. Imports (first code cell)
# ---------------------------------------------------------------------------
cells.append(code(r'''
# First code cell: import everything the notebook needs (offline, installed packages only).
from pathlib import Path
DATA = Path("../../data")  # datasets live two levels above this notebook

# Standard library: OOP helpers, exceptions, files, regex, databases.
import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import tempfile
import csv
import json
import re
import sqlite3

# SQLAlchemy 2.0 Core + ORM (SQLite in-memory; no server).
import sqlalchemy
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float,
    insert, select, update, delete, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

print("Python + libraries ready. SQLAlchemy", sqlalchemy.__version__)
'''))

# ===========================================================================
# CLASSES AND OBJECTS
# ===========================================================================
cells.append(md(r'''
## 1. Classes and objects

A **class** is a blueprint; an **object** (instance) is one concrete value built from it. A class
groups data (attributes) and behaviour (methods). Defining a class creates a new namespace; calling
the class like a function constructs an instance. Methods are functions stored on the class; Python
automatically passes the instance as the first argument, conventionally named `self`.

### Class vs instance attributes

An attribute defined in the class body is a **class attribute**, shared by every instance until one
shadows it. An attribute assigned to `self` inside `__init__` belongs to that single instance. Looking
up `obj.attr` searches the instance dictionary first, then the class. Use `vars(obj)` to see instance
attributes and `ClassName.__dict__` to see class attributes.
'''))

cells.append(code(r'''
# Minimal classes: construct objects, add attributes, call methods.
class Dog:
    """A simple dog with a name and one behaviour."""

    def bark(self):
        # self is the instance that called the method.
        return f"{self.name} says woof!"

dog_a, dog_b = Dog(), Dog()
dog_a.name = "Rex"          # attribute added to this instance only
dog_b.name = "Luna"
print(dog_a.bark(), "|", dog_b.bark())
print(type(dog_a), isinstance(dog_a, Dog))

# Shared class attribute vs per-instance attribute.
class Student:
    school = "Central High"          # shared by all instances

    def __init__(self, name):
        self.name = name             # unique to each instance

s1, s2 = Student("Ana"), Student("Ben")
print(s1.school, s2.school)          # both read the class attribute
s2.school = "West High"              # shadows it on s2 only
print(s1.school, s2.school)
print("s1 instance dict:", vars(s1))
print("s2 instance dict:", vars(s2))
print("class dict keys:", [k for k in Student.__dict__ if not k.startswith("__")])
'''))

cells.append(md(r'''
### The `__init__` constructor, `__repr__` / `__str__`, and instance methods

`__init__(self, ...)` runs automatically when an object is created; it initialises instance state.
It is an initialiser, not the allocator — `__new__` creates the object and `__init__` fills it.

- `__str__` returns a readable, user-facing description; used by `str(obj)` and `print(obj)`.
- `__repr__` returns an unambiguous, ideally evaluable description; used by `repr(obj)` and in lists.
  If only one is defined, the other falls back to it.

Instance methods are functions whose first parameter is the instance. Calling `obj.method(a)` is
shorthand for `Class.method(obj, a)`. `self` lets methods read and change object state.
'''))

cells.append(code(r'''
# Constructor, both string representations, and state-mutating methods.
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point at ({self.x}, {self.y})"        # friendly form

    def __repr__(self):
        return f"Point({self.x!r}, {self.y!r})"        # unambiguous form

p = Point(3, 4)
print(str(p))
print(repr(p))
print([p])                                             # list display uses __repr__

class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        return self.balance

account = BankAccount("Ana", 100)
print(account.deposit(50), account.withdraw(30), account.balance)
print(BankAccount.withdraw(account, 10))               # explicit self: same as account.withdraw(10)
'''))

cells.append(md(r'''
### `@classmethod`, `@staticmethod`, and `@property`

- `@classmethod` receives the **class** as its first argument (`cls`). Use it for alternative
  constructors and factories that need to know which class was called.
- `@staticmethod` receives no implicit first argument. It behaves like a plain function namespaced
  inside the class — a utility helper that touches neither instance nor class state.
- `@property` gives a method the syntax of an attribute. `@property` defines a getter;
  `@name.setter` defines a setter that can validate assignment. Omitting the setter makes the
  attribute effectively read-only. This is the Pythonic way to encapsulate state.
'''))

cells.append(code(r'''
# classmethod alternative constructors, a staticmethod helper, and validated properties.
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    @classmethod
    def margherita(cls):
        return cls(["tomato", "mozzarella"])           # cls = the calling class

    @classmethod
    def from_string(cls, text):
        return cls([part.strip() for part in text.split(",")])

    @staticmethod
    def oven_temperature(celsius):
        return celsius * 9 / 5 + 32

print(Pizza.margherita().ingredients)
print(Pizza.from_string("ham, pineapple").ingredients)
print(Pizza.oven_temperature(200))

class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9 / 5 + 32                # computed, read-only

t = Temperature(0)
t.celsius = 100                                         # calls the setter
print(t.celsius, t.fahrenheit)
try:
    t.celsius = -300                                    # triggers validation
except ValueError as exc:
    print("rejected:", exc)
'''))

cells.append(md(r'''
### Inheritance: single, multiple, `super()`, overriding

Inheritance lets a child class reuse and extend a parent's behaviour. **Single inheritance** has one
parent; **multiple inheritance** combines several parents. Python resolves ambiguity with the Method
Resolution Order (MRO), a C3 linearisation you can inspect with `Class.mro()`.

- `super()` returns a proxy to the next class in the MRO, used to call the parent's `__init__` or methods.
- **Overriding** means defining a method in the child with the same name as the parent's.
'''))

cells.append(code(r'''
# Single inheritance with super() and overriding.
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)               # initialise the parent part
        self.breed = breed

    def speak(self):                         # override
        return f"{self.name} the {self.breed} barks"

class Cat(Animal):
    def speak(self):
        return f"{self.name} meows"

for pet in (Dog("Rex", "lab"), Cat("Luna")):
    print(pet.speak())
print("Dog MRO:", [c.__name__ for c in Dog.mro()])

# Multiple inheritance follows the MRO: Duck -> Flyer -> Swimmer -> object.
class Flyer:
    def move(self):
        return "flying"

class Swimmer:
    def move(self):
        return "swimming"

class Duck(Flyer, Swimmer):
    def move(self):
        return f"{super().move()} and then {Swimmer.move(self)}"

print(Duck().move())
print("Duck MRO:", [c.__name__ for c in Duck.mro()])
'''))

cells.append(md(r'''
### Polymorphism: duck typing and operator overloading

**Duck typing** judges an object by the methods it provides, not by its declared type: a function
that calls `.area()` works with any object implementing `area`, regardless of its class.

**Operator overloading** uses dunder (double-underscore) methods: `__add__` makes `+` work,
`__len__` makes `len()` work, `__getitem__` enables indexing, `__eq__` defines `==`, and `__lt__`
enables sorting. Reflected variants such as `__radd__` handle the left operand being another type.
'''))

cells.append(code(r'''
# Duck typing: one function accepts several unrelated types.
def describe_shape(shape):
    return f"area = {shape.area():.2f}"

class Square:
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side ** 2

class Circle:
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        import math
        return math.pi * self.radius ** 2

print(describe_shape(Square(3)))
print(describe_shape(Circle(2)))

# Operator overloading with dunder methods.
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.x, self.y)[index]

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1, v2 = Vector(1, 2), Vector(3, 4)
print(v1 + v2)
print(v1 == Vector(1, 2))
print(len(v1), v1[0], list(v1))
'''))

cells.append(md(r'''
### Abstraction with `abc.ABC` and `@abstractmethod`

An **abstract base class** defines an interface that subclasses must implement. Marking a method
with `@abstractmethod` prevents the class itself from being instantiated; a concrete subclass must
override every abstract method. `abc` supports `class C(ABC)` and the `metaclass=ABCMeta` form. This
enforces a contract while still allowing shared concrete methods in the base.
'''))

cells.append(code(r'''
# Abstract base class enforces an interface.
class Shape(ABC):
    @abstractmethod
    def area(self):
        """Subclasses must implement area()."""
        ...

    @abstractmethod
    def perimeter(self):
        ...

    def describe(self):
        # Shared concrete method built on the abstract interface.
        return f"area={self.area():.2f}, perimeter={self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width, self.height = width, height
    def area(self):
        return self.width * self.height
    def perimeter(self):
        return 2 * (self.width + self.height)

print(Rectangle(3, 4).describe())
try:
    Shape()                              # cannot instantiate an abstract class
except TypeError as exc:
    print("abstract:", exc)
print("abstract methods:", Shape.__abstractmethods__)
'''))

cells.append(md(r'''
### Encapsulation and name mangling

**Encapsulation** keeps internal state hidden behind a public interface. Python uses conventions
rather than hard enforcement:

- `_name` (single underscore) signals "internal use" to other programmers.
- `__name` (double leading underscore, no trailing pair) triggers **name mangling**: it is stored as
  `_ClassName__name`, avoiding collisions in subclasses. It is still reachable by that name, so it
  is a convention, not true privacy.
- `__name__` (dunder on both sides) is reserved for Python's special methods — never invent new ones.
'''))

cells.append(code(r'''
# Name mangling demonstrated.
class SecretHolder:
    def __init__(self):
        self.public = "visible"
        self._internal = "by convention"
        self.__hidden = "mangled"        # stored as _SecretHolder__hidden

    def reveal(self):
        return self.__hidden

obj = SecretHolder()
print(obj.public, obj._internal)
print(obj.reveal())
print("instance dict:", vars(obj))
print("accessible via mangled name:", obj._SecretHolder__hidden)
'''))

cells.append(md(r'''
### Dataclasses (`@dataclass`, `field()`) and `__slots__`

`@dataclass` auto-generates `__init__`, `__repr__`, `__eq__` and other dunders from annotated fields,
removing boilerplate for data-holding classes. `field()` customises fields: `default_factory` is the
correct way to give a mutable default (a fresh list per instance); `default`, `repr=False`,
`compare=False` and `kw_only` are other options. `asdict()` converts an instance to a dictionary.

`__slots__` restricts instances to a fixed set of attributes and removes the per-instance `__dict__`.
It saves memory for many objects and prevents accidental attribute creation. Trade-offs: no `__dict__`
by default, no weak references unless `"__weakref__"` is included, and restricted multiple inheritance.
'''))

cells.append(code(r'''
# Dataclass reduces boilerplate; field() avoids shared mutable defaults.
@dataclass
class Book:
    title: str
    author: str
    year: int = 0
    tags: list[str] = field(default_factory=list)   # fresh list per instance
    isbn: str = field(default="", repr=False)       # hidden from repr

b1 = Book("Dune", "Frank Herbert", 1965, ["sci-fi"])
b2 = Book("1984", "George Orwell")
b1.tags.append("classic")
print(b1)
print(b2.tags)                  # still empty: no shared mutable default
print(b1 == Book("Dune", "Frank Herbert", 1965, ["sci-fi"]))
print(asdict(b2))

# __slots__ fixes the attribute set and drops the instance __dict__.
class Point3D:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

pt = Point3D(1, 2, 3)
print(pt.x, pt.y, pt.z, "has __dict__:", hasattr(pt, "__dict__"))
try:
    pt.extra = 9                  # not allowed: not in __slots__
except AttributeError as exc:
    print("blocked:", exc)
'''))

# ---------------------------------------------------------------------------
# TRY IT YOURSELF 1
# ---------------------------------------------------------------------------
cells.append(md(r'''
### Try it yourself #1 — OOP mini challenge

Create an abstract `Shape` base class with an abstract `area()` method, then define `Circle` and
`Rectangle` subclasses. Add a `@property` that validates that dimensions are positive, give the base
a `__str__`, and confirm duck typing by summing the areas of mixed shapes in one list. Write your
version in the next cell, then compare with the solution.
'''))

cells.append(code(r'''
# Your turn: write the Shape hierarchy here before looking at the solution.
'''))

cells.append(md(r'''
#### Solution #1
'''))

cells.append(code(r'''
# Worked solution: abstract base, validated properties, polymorphism.
class Shape(ABC):
    @abstractmethod
    def area(self):
        ...

    def __str__(self):
        return f"{type(self).__name__} with area {self.area():.2f}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("radius must be positive")
        self._radius = value
    def area(self):
        import math
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width, self.height = width, height
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("width must be positive")
        self._width = value
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("height must be positive")
        self._height = value
    def area(self):
        return self.width * self.height

shapes = [Circle(2), Rectangle(3, 4)]
for shape in shapes:
    print(shape)
print("total area:", sum(s.area() for s in shapes))   # duck typing in action
'''))

# ===========================================================================
# EXCEPTIONS
# ===========================================================================
cells.append(md(r'''
## 2. Exception handling

An exception is an error signalled at runtime. `try` runs risky code; `except` catches a specific
type; `else` runs only when no exception occurred; `finally` always runs, even on early `return` or
`break` — ideal for cleanup. Catch specific exceptions, not bare `except`, so real bugs are not hidden.

Use `raise` to signal an error yourself. `raise ValueError("message")` raises a built-in; `raise`
alone re-raises the active exception inside an `except` block. Define custom exceptions by subclassing
`Exception` (or a specific built-in). Chaining with `raise ... from ...` preserves the original cause;
`__context__` and `__cause__` record it. Common built-ins: `ValueError`, `TypeError`, `KeyError`,
`IndexError`, `AttributeError`, `FileNotFoundError`, `ZeroDivisionError`, and `StopIteration`
(raised by iterators when exhausted).
'''))

cells.append(code(r'''
# try / except / else / finally in full.
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError as exc:
        print("cannot divide by zero:", exc)
        return None
    except TypeError as exc:
        print("operands must be numbers:", exc)
        return None
    else:
        print("division succeeded")
        return result
    finally:
        print("finally always runs")

print(divide(10, 2))
print(divide(10, 0))
'''))

cells.append(code(r'''
# Custom exceptions, raising, re-raising with a chained cause, and built-ins.
class InsufficientFunds(Exception):
    """Raised when a withdrawal exceeds the available balance."""

    def __init__(self, balance, requested):
        super().__init__(f"requested {requested}, balance is {balance}")
        self.balance = balance
        self.requested = requested

def withdraw(balance, amount):
    try:
        if amount > balance:
            raise InsufficientFunds(balance, amount)
    except InsufficientFunds as exc:
        print("caught:", exc)
        raise RuntimeError("withdrawal failed") from exc   # explicit chaining
    return balance - amount

try:
    withdraw(50, 80)
except RuntimeError as exc:
    print("outer:", exc)
    print("cause:", type(exc.__cause__).__name__)

for action in (lambda: int("abc"), lambda: {}["missing"], lambda: [][0]):
    try:
        action()
    except Exception as exc:
        print(type(exc).__name__, "->", exc)
'''))

cells.append(md(r'''
### Context managers and the `with` statement

A context manager sets up and reliably tears down a resource. The protocol is `__enter__` (returns
the resource) and `__exit__(exc_type, exc_val, exc_tb)` (cleans up; returning `True` suppresses an
exception). `with` guarantees `__exit__` runs even if the body raises. Files, locks and DB connections
are context managers. For quick custom ones, use `@contextmanager` from `contextlib` and a single
`yield` — code before `yield` is setup, code after is teardown.
'''))

cells.append(code(r'''
# Two ways to build a context manager.
class Timer:
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self.start
        print(f"elapsed {self.elapsed:.6f}s; exception={exc_type}")
        return False        # do not suppress exceptions

@contextmanager
def managed_resource(name):
    print("setup:", name)
    try:
        yield name          # the value bound by `as`
    finally:
        print("teardown:", name)

with Timer() as timer:
    _ = sum(range(1000))

with managed_resource("connection") as resource:
    print("using", resource)
'''))

# ===========================================================================
# FILES
# ===========================================================================
cells.append(md(r'''
## 3. Working with files

`open(path, mode)` returns a file object. Modes: `r` read (default), `w` write (truncates), `a` append,
`x` exclusive create; add `b` for binary (`rb`, `wb`) and `+` for read/write, and pass
`encoding="utf-8"` explicitly. Text methods: `read()` (all), `readline()` (one line), `readlines()`
(list of lines); `write()` and `writelines()` for output. `tell()` gives the byte offset and
`seek(offset, whence)` moves it (`whence`: 0 start, 1 current, 2 end). Using `with` closes the file
automatically. Demos use `tempfile` so no files are written into the project.
'''))

cells.append(code(r'''
# Text file read/write, seek/tell, readline/readlines.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "notes.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("first line\nsecond line\nthird line\n")
        f.writelines(["fourth line\n", "fifth line\n"])

    with open(path, "r", encoding="utf-8") as f:
        print("mode/name:", f.mode, f.name)
        print("tell at open:", f.tell())
        print("readline:", f.readline().strip())
        print("tell after one line:", f.tell())
        print("readlines:", [line.strip() for line in f.readlines()])

    with open(path, "r", encoding="utf-8") as f:
        f.seek(0, 2)                      # end of file
        print("size via seek:", f.tell())
        f.seek(0)                         # back to start
        print("read all:\n" + f.read())
'''))

cells.append(md(r'''
### `pathlib.Path`

`Path` offers an object-oriented filesystem API replacing many `os.path` calls. Build paths with
`/`, inspect with `.name`, `.stem`, `.suffix`, `.parent`, `.parts`; test with `.exists()`, `.is_file()`,
`.is_dir()`; create with `.mkdir(parents=True, exist_ok=True)`; read/write with `.read_text()` and
`.write_text()`; iterate with `.iterdir()` and `.glob("*.csv")`. `.resolve()` returns an absolute path.
The `DATA = Path("../../data")` helper at the top points to the shared datasets folder.
'''))

cells.append(code(r'''
# pathlib.Path essentials.
p = DATA / "iris.csv"
print("path:", p)
print("name/stem/suffix:", p.name, p.stem, p.suffix)
print("parent:", p.parent)
print("exists:", p.exists())
print("absolute:", p.resolve())

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "a.csv").write_text("x\n", encoding="utf-8")
    (root / "b.csv").write_text("y\n", encoding="utf-8")
    print("csv files:", [f.name for f in root.glob("*.csv")])
    print("read back:", (root / "a.csv").read_text(encoding="utf-8").strip())
'''))

cells.append(md(r'''
### CSV with the `csv` module and JSON with `json`

`csv.reader` yields each row as a list of strings; `csv.writer` writes rows from sequences.
`csv.DictReader` uses the header row to produce dictionaries, and `csv.DictWriter` writes dictionaries
with a given `fieldnames`. Open with `newline=""` so the module handles line endings. Options include
`delimiter=";"`, `quotechar` and `quoting`.

For JSON: `json.dumps(obj)` serialises to a string, `json.loads(text)` parses back;
`json.dump(obj, file)` and `json.load(file)` work with files. Supported types are `dict`, `list`,
`str`, `int`, `float`, `bool` and `None`; keys must be strings. Useful parameters: `indent=2`,
`sort_keys=True`, `ensure_ascii=False`, and `default=` for custom objects.
'''))

cells.append(code(r'''
# csv module: writer/reader and DictWriter/DictReader.
rows = [["name", "score"], ["Ana", "90"], ["Ben", "85"]]

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            print("row:", row)

    records = [{"name": "Ana", "score": 90}, {"name": "Ben", "score": 85}]
    dpath = Path(tmp) / "scores_dict.csv"
    with open(dpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "score"])
        writer.writeheader()
        writer.writerows(records)

    with open(dpath, "r", newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            print("record:", record, "score+1 =", int(record["score"]) + 1)

# json module: dumps/loads and file dump/load.
data = {"name": "Ana", "scores": [90, 85], "active": True, "note": None}
text = json.dumps(data, indent=2, sort_keys=True)
print(text)
print("round trip equal:", json.loads(text) == data)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "data.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    with open(path, "r", encoding="utf-8") as f:
        print("loaded from file:", json.load(f))
'''))

# ===========================================================================
# REGULAR EXPRESSIONS
# ===========================================================================
cells.append(md(r'''
## 4. Regular expressions with `re`

A regular expression is a mini-language for matching text.

- **Literals and classes**: `.` any char (except newline), `\d` digit, `\D` non-digit, `\w` word char,
  `\W` non-word, `\s` whitespace, `\S` non-whitespace; `[abc]` one of a set, `[^abc]` negated,
  `[a-z]` range; escapes `\.` `\n` `\t`.
- **Quantifiers**: `*` zero or more, `+` one or more, `?` zero or one, `{n}` exactly n, `{m,n}` range,
  `{n,}` at least n. Add `?` after a quantifier for non-greedy matching (`.*?`).
- **Anchors**: `^` start of string (or line with `re.M`), `$` end, `\b` word boundary, `\A` start only,
  `\Z` end only.
- **Groups**: `(...)` capturing, `(?:...)` non-capturing, `(?P<name>...)` named, `a|b` alternation,
  backreference `\1`.
- **Raw strings**: write patterns as `r"\d+"` so backslashes are not consumed by Python escaping.
- **Flags**: `re.I`/`IGNORECASE`, `re.M`/`MULTILINE`, `re.S`/`DOTALL`, `re.X`/`VERBOSE`, `re.A`/`ASCII`;
  combine with `|` or pass `flags=`.

Functions: `re.search` (first match anywhere), `re.match` (only at start), `re.fullmatch` (whole string),
`re.findall` (list), `re.finditer` (iterator of match objects), `re.sub` (replace), `re.split` (split),
`re.compile` (reuse). Match objects expose `.group()`, `.groups()`, `.start()`, `.end()`, `.span()`.
'''))

cells.append(code(r'''
# Core re functions: search, match, fullmatch, findall, finditer.
text = "Contact ana@example.com or ben@example.org today."
pattern = r"[\w.+-]+@[\w.-]+"          # simple email-shaped pattern

print("search:", re.search(pattern, text).group())
print("match at start:", re.match(pattern, text))          # None: starts with 'Contact'
print("fullmatch:", re.fullmatch(r"\d+", "12345"))
print("fullmatch fails:", re.fullmatch(r"\d+", "12a"))
print("findall:", re.findall(pattern, text))
for m in re.finditer(pattern, text):
    print("match", m.group(), "span", m.span())

# Groups, alternation, quantifiers, anchors, and flags.
log = "ERROR 404 not found; WARN 12 low memory; ERROR 500 server error"
rx = re.compile(r"(?P<level>ERROR|WARN)\s+(?P<code>\d+)")
for m in rx.finditer(log):
    print(m.group("level"), m.group("code"), m.groupdict())

print("quantifiers:", re.findall(r"\d{2,3}", "1 12 123 1234"))
print("anchored:", bool(re.search(r"^Hello", "Hello world")))
print("ignorecase:", re.findall(r"python", "Python PYTHON", flags=re.I))
print("multiline:", re.findall(r"^\w+", "a\nb\nc", flags=re.M))

verbose = re.compile(r"""
    \b            # word boundary
    \d{4}         # year
    -             # separator
    \d{2}         # month
""", re.X)
print("verbose:", verbose.findall("date 2026-01 and 1999-12"))
'''))

cells.append(md(r'''
### Common patterns: `sub`, `split`, email/phone/date

`re.sub(pattern, repl, text)` replaces matches (`repl` can be a string or a function); `count=`
limits replacements. `re.split(pattern, text)` splits on matches; capturing groups are included in
the result, so use `(?:...)` to suppress them. Frequently needed patterns: an email like
`[\w.+-]+@[\w-]+\.[\w.-]+`, a phone like `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`, and an ISO date like
`\d{4}-\d{2}-\d{2}`. Validate whole fields with `fullmatch`, not `search`.
'''))

cells.append(code(r'''
# sub/split and common validation patterns.
email_rx = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
phone_rx = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
date_rx = re.compile(r"\d{4}-\d{2}-\d{2}")

text = "Mail ana@mail.com or call (555) 123-4567 on 2026-11-18."
print("emails:", email_rx.findall(text))
print("phones:", phone_rx.findall(text))
print("dates:", date_rx.findall(text))
print("valid email:", bool(email_rx.fullmatch("ana@example.com")))
print("invalid email:", bool(email_rx.fullmatch("not-an-email")))

print("sub:", re.sub(r"\d+", "#", "a1 b22 c333"))
print("sub with function:", re.sub(r"\d+", lambda m: str(int(m.group()) * 2), "1 2 3"))
print("split:", re.split(r"\s*,\s*", "a, b ,c"))
print("split with groups:", re.split(r"(\d+)", "a1b2"))
print("split no groups:", re.split(r"(?:\d+)", "a1b2"))
'''))

# TRY IT YOURSELF 2
cells.append(md(r'''
### Try it yourself #2 — Regex challenge

Given the string below, write one pattern that extracts every ISO date (`YYYY-MM-DD`) and use
`re.sub` to mask any email address as `[EMAIL]`. Also validate a phone number with `fullmatch`.
Try it in the next cell before reading the solution.
'''))

cells.append(code(r'''
# Your turn: solve the regex challenge here.
sample = "On 2026-01-15 ana@mail.com wrote; on 2026-02-28 bob@site.org followed."
'''))

cells.append(md(r'''
#### Solution #2
'''))

cells.append(code(r'''
# Worked solution #2.
sample = "On 2026-01-15 ana@mail.com wrote; on 2026-02-28 bob@site.org followed."
date_rx = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
email_rx = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
phone_rx = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

print("dates:", date_rx.findall(sample))
print("masked:", email_rx.sub("[EMAIL]", sample))
print("valid phone:", bool(phone_rx.fullmatch("(555) 123-4567")))
print("invalid phone:", bool(phone_rx.fullmatch("12345")))
'''))

# ===========================================================================
# SQL AND DATABASES
# ===========================================================================
cells.append(md(r'''
## 5. SQL and database access

A **relational database** stores data in tables of rows and columns. A **primary key** uniquely
identifies a row; a **foreign key** references a row in another table. Core SQL statements:

- `CREATE TABLE name (col TYPE [constraints], ...)` defines structure.
- `INSERT INTO name (cols) VALUES (...)` adds rows.
- `SELECT cols FROM table WHERE cond GROUP BY col HAVING cond ORDER BY col [ASC|DESC] LIMIT n` queries.
- `UPDATE table SET col=value WHERE cond` and `DELETE FROM table WHERE cond` change data — always use `WHERE`.
- `JOIN` combines tables: `INNER JOIN` (matching rows), `LEFT JOIN` (all left rows), etc.
- Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, often with `GROUP BY`.

We run raw SQL with `sqlite3` in memory (`:memory:`), then use SQLAlchemy 2.0 Core and ORM, also
against in-memory SQLite. MySQL/PostgreSQL examples appear only as non-executed templates.
'''))

cells.append(code(r'''
# sqlite3 in-memory: full CRUD, WHERE, GROUP BY, ORDER BY, JOIN, aggregates.
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.execute("""CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)""")
cur.execute("""CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    salary REAL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
)""")

cur.executemany("INSERT INTO departments (id, name) VALUES (?, ?)",
                [(1, "Engineering"), (2, "Sales")])
cur.executemany(
    "INSERT INTO employees (name, department_id, salary) VALUES (?, ?, ?)",
    [("Ana", 1, 90000), ("Ben", 1, 80000), ("Cara", 2, 70000), ("Dan", 2, 75000)],
)
conn.commit()

query = """SELECT d.name AS dept, COUNT(*) AS headcount, ROUND(AVG(e.salary), 2) AS avg_salary
           FROM employees e
           JOIN departments d ON e.department_id = d.id
           WHERE e.salary >= 60000
           GROUP BY d.name
           ORDER BY avg_salary DESC"""
print("grouped results:")
for row in cur.execute(query).fetchall():
    print("  ", row)

cur.execute("UPDATE employees SET salary = salary * 1.10 WHERE department_id = 1")
cur.execute("DELETE FROM employees WHERE name = ?", ("Dan",))
conn.commit()
print("after update/delete:", cur.execute("SELECT name, salary FROM employees ORDER BY name").fetchall())
conn.close()
'''))

cells.append(md(r'''
### SQLAlchemy 2.0 Core

SQLAlchemy **Core** provides a SQL expression language that builds portable statements in Python and
executes them through an `Engine`/`Connection`. Define tables with `MetaData` + `Table` + `Column`;
use `insert()`, `select()`, `update()`, `delete()` and `func` instead of raw strings. In 2.0 style,
DML runs via `connection.execute(stmt)` and transactions are committed explicitly with
`connection.commit()` ("commit as you go"). `conn.execute(stmt).all()` returns rows whose fields are
accessible by name.
'''))

cells.append(code(r'''
# SQLAlchemy 2.0 Core with in-memory SQLite.
engine = create_engine("sqlite:///:memory:", echo=False)
metadata = MetaData()

employees = Table(
    "employees_core", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50), nullable=False),
    Column("department", String(50)),
    Column("salary", Float),
)
metadata.create_all(engine)

with engine.connect() as conn:
    conn.execute(insert(employees), [
        {"name": "Ana", "department": "Eng", "salary": 90000},
        {"name": "Ben", "department": "Eng", "salary": 80000},
        {"name": "Cara", "department": "Sales", "salary": 70000},
    ])
    conn.commit()  # 2.0: explicitly commit DML

with engine.connect() as conn:
    stmt = (select(employees.c.name, employees.c.salary)
            .where(employees.c.salary >= 75000)
            .order_by(employees.c.salary.desc()))
    print("filtered:", [(row.name, row.salary) for row in conn.execute(stmt)])

    agg = (select(employees.c.department, func.count().label("n"),
                  func.avg(employees.c.salary).label("avg_salary"))
           .group_by(employees.c.department))
    print("aggregates:", [(r.department, r.n, round(r.avg_salary, 2))
                          for r in conn.execute(agg)])

    conn.execute(update(employees).where(employees.c.name == "Ben").values(salary=85000))
    conn.execute(delete(employees).where(employees.c.name == "Cara"))
    conn.commit()

with engine.connect() as conn:
    print("after changes:", conn.execute(select(employees.c.name, employees.c.salary)).all())
'''))

cells.append(md(r'''
### SQLAlchemy 2.0 ORM

The **ORM** maps Python classes to tables so you work with objects instead of rows. In 2.0 style,
declare a `DeclarativeBase` subclass, then model classes with `Mapped[...]` annotations and
`mapped_column(...)`. Persist with a `Session`: `session.add_all(...)`, `session.commit()`, then query
with `select(Model).where(...)` and `.scalars().all()`. The session handles transactions and refreshes
instances after commit. The declarative base keeps a shared `MetaData` via `Base.metadata`.
'''))

cells.append(code(r'''
# SQLAlchemy 2.0 ORM with in-memory SQLite.
class Base(DeclarativeBase):
    pass

class Employee(Base):
    __tablename__ = "employees_orm"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    department: Mapped[str | None] = mapped_column(String(50), nullable=True)
    salary: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self):
        return f"Employee({self.name!r}, {self.department!r}, {self.salary})"

orm_engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(orm_engine)

with Session(orm_engine) as session:
    session.add_all([
        Employee(name="Ana", department="Eng", salary=90000),
        Employee(name="Ben", department="Eng", salary=80000),
        Employee(name="Cara", department="Sales", salary=70000),
    ])
    session.commit()

    stmt = select(Employee).where(Employee.salary >= 80000).order_by(Employee.salary)
    print("ORM results:", session.execute(stmt).scalars().all())

    ben = session.execute(select(Employee).where(Employee.name == "Ben")).scalar_one()
    ben.salary = 82000                  # change is tracked by the session
    session.commit()
    print("after update:", ben)
'''))

# TRY IT YOURSELF 3
cells.append(md(r'''
### Try it yourself #3 — SQL challenge

Using `sqlite3` in memory, create a `products(id, name, category, price)` table, insert at least five
products across two categories, then write ONE query that returns each category with its product count
and the most expensive product's name. HINT: `GROUP BY`, `COUNT`, and a correlated subquery.
Try it below, then check the solution.
'''))

cells.append(code(r'''
# Your turn: solve the SQL challenge here.
'''))

cells.append(md(r'''
#### Solution #3
'''))

cells.append(code(r'''
# Worked solution #3.
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("""CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
)""")
cur.executemany("INSERT INTO products (name, category, price) VALUES (?, ?, ?)",
                [("Laptop", "Electronics", 1200), ("Mouse", "Electronics", 25),
                 ("Monitor", "Electronics", 300), ("Desk", "Furniture", 450),
                 ("Chair", "Furniture", 200)])

query = """SELECT category,
                  COUNT(*) AS n_products,
                  (SELECT p2.name FROM products p2
                   WHERE p2.category = p.category
                   ORDER BY p2.price DESC LIMIT 1) AS priciest
           FROM products p
           GROUP BY category
           ORDER BY category"""
for row in cur.execute(query).fetchall():
    print(row)
conn.close()
'''))

cells.append(md(r'''
### MySQL and PostgreSQL templates (need a running server — do NOT run here)

These are **templates only**, shown as fenced code so they cannot break top-to-bottom execution.
They require a running MySQL or PostgreSQL server and valid credentials. The pattern is the same:
build a SQLAlchemy URL, `create_engine`, then use Core or ORM exactly as above. Install drivers with
`pip install pymysql` (MySQL) or `pip install psycopg2-binary` (PostgreSQL).

**MySQL** (`pymysql`, URL `mysql+pymysql://user:password@host:3306/dbname`):

```python
# TEMPLATE — requires a MySQL server. Not executed in this notebook.
import pymysql
from sqlalchemy import create_engine, text

url = "mysql+pymysql://user:password@localhost:3306/mydb"
engine = create_engine(url, echo=False)
with engine.connect() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name VARCHAR(50))"))
    conn.execute(text("INSERT INTO users (id, name) VALUES (1, 'Ana')"))
    conn.commit()
    print(conn.execute(text("SELECT * FROM users")).all())
```

**PostgreSQL** (`psycopg2`, URL `postgresql+psycopg2://user:password@host:5432/dbname`):

```python
# TEMPLATE — requires a PostgreSQL server. Not executed in this notebook.
import psycopg2
from sqlalchemy import create_engine, text

url = "postgresql+psycopg2://user:password@localhost:5432/mydb"
engine = create_engine(url, echo=False)
with engine.connect() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, name VARCHAR(50))"))
    conn.execute(text("INSERT INTO users (name) VALUES ('Ana')"))
    conn.commit()
    print(conn.execute(text("SELECT * FROM users")).all())
```

Differences: MySQL uses `AUTO_INCREMENT`, PostgreSQL uses `SERIAL`/`IDENTITY`; default ports are 3306
and 5432; keep credentials out of source code and use environment variables or a secret manager.
'''))

# ===========================================================================
# CHEAT SHEET
# ===========================================================================
cells.append(md(r'''
## Cheat sheet / key takeaways

| Topic | Essentials |
| --- | --- |
| Classes | `class C:`, `__init__` initialises, `self` is the instance, `obj.attr` checks instance then class. |
| Attributes | Class attributes are shared; instance attributes (`self.x`) are per-object; `vars(obj)` lists them. |
| Dunders | `__repr__` unambiguous, `__str__` readable, `__len__`, `__getitem__`, `__add__`, `__eq__`, `__lt__`. |
| Method kinds | `@classmethod` gets `cls` (alt constructors), `@staticmethod` no implicit arg, `@property` + setter validates. |
| Inheritance | `class C(Base)`, `super()` calls parent, override by redefining, `C.mro()` shows order. |
| Polymorphism | Duck typing (any object with the needed method) + operator overloading via dunders. |
| Abstraction | `class C(ABC)` + `@abstractmethod` enforces a contract; abstract classes cannot be instantiated. |
| Encapsulation | `_x` convention, `__x` name-mangled to `_Class__x`, `__x__` reserved for Python. |
| Dataclasses | `@dataclass` generates `__init__`/`__repr__`/`__eq__`; `field(default_factory=list)` for mutable defaults. |
| `__slots__` | Fixed attribute set, no per-instance `__dict__`, lower memory. |
| Exceptions | `try/except/else/finally`, `raise`, custom `Exception` subclass, `raise ... from ...` to chain. |
| Context managers | `with`, `__enter__`/`__exit__`, `@contextmanager` + `yield`, guaranteed cleanup. |
| Files | `open(mode='r/w/a/x/b/+')`, `read/line(s)`, `write/writelines`, `seek/tell`, always use `with`. |
| pathlib | `Path`, `/`, `.name/.stem/.suffix/.parent`, `.glob()`, `.read_text()/.write_text()`. |
| CSV / JSON | `csv.reader/DictReader`, `DictWriter`, `newline=""`; `json.dumps/loads/dump/load`, `indent`, `sort_keys`. |
| Regex | raw strings `r"..."`, classes `\d\w\s`, quantifiers `*+?{n,m}`, anchors `^$\b`, groups `(...)`, flags `re.I/M/S/X`. |
| re functions | `search`, `match`, `fullmatch`, `findall`, `finditer`, `sub`, `split`, `compile`; match `.group/.span`. |
| SQL | `CREATE/INSERT/SELECT/WHERE/GROUP BY/HAVING/ORDER BY/JOIN/UPDATE/DELETE`, aggregates `COUNT/SUM/AVG`. |
| sqlite3 | `connect(":memory:")`, `cursor()`, `execute/executemany`, `commit()`, parameterised `?`. |
| SQLAlchemy 2.0 | `create_engine`, Core `Table/Column`, `select()/insert()/update()/delete()`, `commit()`; ORM `DeclarativeBase`, `Mapped`, `mapped_column`, `Session`. |
| RDBMS | MySQL `mysql+pymysql://...` port 3306; PostgreSQL `postgresql+psycopg2://...` port 5432 — need a server. |

Three habits prevent most bugs: use `with` for resources, use parameterised SQL (never string
formatting), and catch specific exceptions rather than bare `except`.
'''))

cells.append(md(r'''
## Further reading

- [Python 3 reference — Classes (full)](https://docs.python.org/3/reference/classes.html)
- [Python 3 — Data model (dunder methods)](https://docs.python.org/3/reference/datamodel.html)
- [Python 3 — `functools`](https://docs.python.org/3/library/functools.html)
- [Python 3 — `typing`](https://docs.python.org/3/library/typing.html)
- [Python 3 — `contextlib`](https://docs.python.org/3/library/contextlib.html)
- [Python 3 — Regular expression HOWTO](https://docs.python.org/3/howto/regex.html)
- [Python 3 — `re` module reference](https://docs.python.org/3/library/re.html)
- [SQLite — SQL syntax reference](https://www.sqlite.org/lang.html)
- [SQLAlchemy 2.0 — SQL expression language (Core)](https://docs.sqlalchemy.org/en/20/core/)
- [SQLAlchemy 2.0 — ORM](https://docs.sqlalchemy.org/en/20/orm/)
- [Psycopg 2 documentation](https://www.psycopg.org/docs/)
- [PyMySQL documentation](https://pymysql.readthedocs.io/)
- [PostgreSQL documentation](https://www.postgresql.org/docs/)
- [MySQL documentation](https://dev.mysql.com/doc/)
'''))

write_notebook(
    "notebooks/python-fundamentals/unit3_oops_files_regex_sql.ipynb",
    cells,
)
