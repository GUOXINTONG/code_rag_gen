"""
Sample documents for testing the Level-3 Question Generator.
These represent different types of programming documentation that would be used
as reference material for generating synthetic coding questions.
"""

# Sample Document 1: Data Structures
DATA_STRUCTURES_DOC = """
# Data Structures in Python

## Arrays and Lists
Arrays are fundamental data structures that store elements in contiguous memory locations. 
In Python, lists provide dynamic array functionality with O(1) average-case append operations.

```python
# Dynamic array operations
arr = [1, 2, 3, 4, 5]
arr.append(6)  # O(1) average case
arr.insert(0, 0)  # O(n) worst case
arr.pop()  # O(1)
arr.pop(0)  # O(n)
```

Key characteristics:
- Random access: O(1)
- Search: O(n)
- Insertion/Deletion: O(n) worst case, O(1) at end

## Linked Lists
Linked lists consist of nodes where each node contains data and a reference to the next node.
They provide efficient insertion and deletion but sacrifice random access.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete_node(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        if current.next:
            current.next = current.next.next
```

## Binary Trees
Binary trees are hierarchical data structures where each node has at most two children.
Binary Search Trees (BST) maintain the property that left child < parent < right child.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        return node
    
    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(node.val)
            self.inorder_traversal(node.right)
```

## Hash Tables
Hash tables provide average O(1) lookup, insertion, and deletion through key-value mapping.
They use hash functions to map keys to array indices.

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)
```

Time complexities:
- Average case: O(1) for all operations
- Worst case: O(n) when all keys hash to same bucket
"""

# Sample Document 2: Algorithms
ALGORITHMS_DOC = """
# Algorithm Design Patterns

## Divide and Conquer
Divide and conquer algorithms solve problems by breaking them into smaller subproblems,
solving each recursively, and combining the results.

### Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

Time Complexity: O(n log n)
Space Complexity: O(n)

### Quick Sort
```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

## Dynamic Programming
Dynamic programming solves complex problems by breaking them down into simpler subproblems
and storing the results to avoid redundant calculations.

### Fibonacci with Memoization
```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Tabulation approach
def fibonacci_tab(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

### Longest Common Subsequence
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

## Greedy Algorithms
Greedy algorithms make locally optimal choices at each step, hoping to find a global optimum.

### Activity Selection Problem
```python
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish_time = activities[0][1]
    
    for start, finish in activities[1:]:
        if start >= last_finish_time:
            selected.append((start, finish))
            last_finish_time = finish
    
    return selected
```

## Graph Algorithms

### Depth-First Search (DFS)
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited
```

### Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

### Dijkstra's Algorithm
```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```
"""

# Sample Document 3: Object-Oriented Programming
OOP_DOC = """
# Object-Oriented Programming Concepts

## Classes and Objects
Classes are blueprints for creating objects. Objects are instances of classes that contain
data (attributes) and functions (methods) that operate on that data.

```python
class Car:
    # Class variable
    wheels = 4
    
    def __init__(self, make, model, year):
        # Instance variables
        self.make = make
        self.model = model
        self.year = year
        self.mileage = 0
    
    def drive(self, miles):
        self.mileage += miles
        print(f"Drove {miles} miles. Total mileage: {self.mileage}")
    
    def __str__(self):
        return f"{self.year} {self.make} {self.model}"

# Creating objects
car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2019)
```

## Inheritance
Inheritance allows a class to inherit attributes and methods from another class,
promoting code reuse and establishing an "is-a" relationship.

```python
class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
    
    def start_engine(self):
        print("Engine started!")
    
    def stop_engine(self):
        print("Engine stopped!")

class Car(Vehicle):
    def __init__(self, make, model, year, doors):
        super().__init__(make, model, year)
        self.doors = doors
    
    def honk(self):
        print("Beep beep!")

class Motorcycle(Vehicle):
    def __init__(self, make, model, year, cc):
        super().__init__(make, model, year)
        self.cc = cc
    
    def wheelie(self):
        print("Doing a wheelie!")
```

## Polymorphism
Polymorphism allows objects of different types to be treated as instances of the same type
through a common interface.

```python
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class Bird(Animal):
    def make_sound(self):
        return "Tweet!"

# Polymorphic behavior
animals = [Dog(), Cat(), Bird()]
for animal in animals:
    print(animal.make_sound())  # Each calls their own implementation
```

## Encapsulation
Encapsulation restricts access to certain components of an object and prevents
the accidental modification of data.

```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self._account_number = account_number  # Protected
        self.__balance = initial_balance       # Private
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}. New balance: ${self.__balance}")
        else:
            print("Deposit amount must be positive")
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid withdrawal amount")
    
    def get_balance(self):
        return self.__balance
    
    @property
    def account_number(self):
        return self._account_number
```

## Abstract Classes and Interfaces
Abstract classes define a common interface but cannot be instantiated directly.
Subclasses must implement abstract methods.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    def description(self):
        return f"This is a {self.__class__.__name__}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius
```

## Design Patterns

### Singleton Pattern
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.data = {}
```

### Factory Pattern
```python
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        elif animal_type.lower() == "bird":
            return Bird()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")
```

### Observer Pattern
```python
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer:
    def __init__(self, name):
        self.name = name
    
    def update(self, message):
        print(f"{self.name} received: {message}")
```
"""

# Sample Document 4: Advanced Python Concepts
ADVANCED_PYTHON_DOC = """
# Advanced Python Programming Concepts

## Decorators
Decorators are a way to modify or enhance functions and classes without permanently modifying them.

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@timer
@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Context Managers
Context managers define runtime context for executing code blocks, commonly used with the `with` statement.

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing file {self.filename}")
        if self.file:
            self.file.close()

# Usage
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")

# Using contextlib
from contextlib import contextmanager

@contextmanager
def database_transaction():
    print("Beginning transaction")
    try:
        yield "db_connection"
        print("Committing transaction")
    except Exception as e:
        print(f"Rolling back transaction: {e}")
        raise
```

## Generators and Iterators
Generators are functions that return an iterator object, yielding items one at a time.

```python
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Generator expression
squares = (x**2 for x in range(10))

# Custom iterator
class NumberSequence:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start >= self.stop:
            raise StopIteration
        current = self.start
        self.start += 1
        return current
```

## Metaclasses
Metaclasses are classes whose instances are classes themselves.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected to database"
    
    def query(self, sql):
        return f"Executing: {sql}"

# Both instances are the same object
db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

## Async Programming
Asynchronous programming allows concurrent execution of code without blocking.

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Async generator
async def async_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        await asyncio.sleep(0.1)  # Simulate async work
        a, b = b, a + b

# Usage
async def main():
    async for num in async_fibonacci(10):
        print(num)

# asyncio.run(main())
```

## Functional Programming
Python supports functional programming paradigms with higher-order functions.

```python
from functools import reduce, partial
from operator import add, mul

# Higher-order functions
def apply_operation(func, *args):
    return func(*args)

def create_multiplier(factor):
    return lambda x: x * factor

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map: apply function to each element
squared = list(map(lambda x: x**2, numbers))
doubled = list(map(create_multiplier(2), numbers))

# Filter: select elements that meet condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
odds = list(filter(lambda x: x % 2 == 1, numbers))

# Reduce: combine all elements into single value
sum_all = reduce(add, numbers)
product_all = reduce(mul, numbers)

# Partial application
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27
```

## Type Hints and Annotations
Type hints provide static type information for better code documentation and IDE support.

```python
from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic

def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_items(items: List[int]) -> Dict[str, int]:
    return {
        "count": len(items),
        "sum": sum(items),
        "average": sum(items) // len(items) if items else 0
    }

def find_user(user_id: int) -> Optional[Dict[str, str]]:
    # Might return None if user not found
    users = {1: {"name": "Alice", "email": "alice@example.com"}}
    return users.get(user_id)

# Generic types
T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
    
    def is_empty(self) -> bool:
        return len(self._items) == 0

# Function that takes another function
def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))
```
"""

# Collect all sample documents
SAMPLE_DOCUMENTS = [
    DATA_STRUCTURES_DOC,
    ALGORITHMS_DOC,
    OOP_DOC,
    ADVANCED_PYTHON_DOC
]

# Document metadata for reference
DOCUMENT_METADATA = [
    {
        "title": "Data Structures in Python",
        "categories": ["data_structure", "implementation"],
        "complexity": "basic_to_intermediate",
        "key_concepts": ["arrays", "linked_lists", "trees", "hash_tables"]
    },
    {
        "title": "Algorithm Design Patterns", 
        "categories": ["algorithm", "problem_solving"],
        "complexity": "intermediate_to_advanced",
        "key_concepts": ["divide_conquer", "dynamic_programming", "greedy", "graph_algorithms"]
    },
    {
        "title": "Object-Oriented Programming",
        "categories": ["design_pattern", "programming_paradigm"],
        "complexity": "basic_to_intermediate", 
        "key_concepts": ["inheritance", "polymorphism", "encapsulation", "abstraction"]
    },
    {
        "title": "Advanced Python Concepts",
        "categories": ["advanced_features", "language_specific"],
        "complexity": "advanced",
        "key_concepts": ["decorators", "generators", "metaclasses", "async_programming"]
    }
]

def get_sample_documents():
    """Return all sample documents as a list."""
    return SAMPLE_DOCUMENTS

def get_document_by_index(index: int) -> str:
    """Get a specific document by index."""
    if 0 <= index < len(SAMPLE_DOCUMENTS):
        return SAMPLE_DOCUMENTS[index]
    raise IndexError(f"Document index {index} out of range")

def get_document_metadata():
    """Return metadata about all sample documents."""
    return DOCUMENT_METADATA

if __name__ == "__main__":
    print(f"Sample documents loaded: {len(SAMPLE_DOCUMENTS)} documents")
    for i, metadata in enumerate(DOCUMENT_METADATA):
        print(f"{i+1}. {metadata['title']} - {metadata['complexity']}")