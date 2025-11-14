---
layout: post
title: "Python Data Structures"
date: 2025-06-26
categories: [Programming, Data Structure]
tags: [python, data structure, stack, queue, tree, hash-table, graph, algorithms, adt]
math: true
---

## [Data Structure] Data Structure

### 0. Types of Python Data Structures

| Term                                                                       | Description                                |
| -------------------------------------------------------------------------- | ------------------------------------------ |
| `list`, `tuple`, `set`, `frozenset`, `dict`, `str`                         | Python built-in data structures            |
| Stack, Queue, Hash Table, Tree, Graph | Abstract Data Types (ADT), require manual implementation |

* Python Built-in Data Structures

| Built-in Data Structure | Description                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| `list`        | Ordered and mutable sequence (dynamic array: automatically adjusts size as needed) |
| `tuple`       | Ordered and immutable sequence                                             |
| `set`         | Unordered collection without duplicates, mutable                                                 |
| `frozenset`   | Immutable set                                                         |
| `dict`        | Hash-based mapping of key-value pairs                                                  |
| `str`         | Immutable string sequence                                                |

* Abstract Data Structures

| Data Structure Name           | Linear / Non-linear | Brief Description                                              |
| ----------------------- | ------------- | -------------------------------------------------------- |
| Stack            | Linear     | Stores and retrieves data in Last-In-First-Out (LIFO) manner           |
| Queue              | Linear     | Stores and retrieves data in First-In-First-Out (FIFO) manner           |
| Tree             | Non-linear   | Hierarchical parent-child relationship structure of nodes               |
| Hash Table | Non-linear   | Structure that uses hash functions to transform keys for fast data access     |
| Graph          | Non-linear   | Generalized structure consisting of nodes (vertices) and connections (edges) between them |

_Note_ : **Data Type** vs **Data Structure**

| Concept                          | Meaning                                                               |
| ----------------------------- | ------------------------------------------------------------------ |
| **Data Type**        | Defines the **kind and size** of values (e.g., `int`, `float`, `str`)          |
| **Data Structure** | Structure for **storing and organizing** data (e.g., `list`, `set`, tree, etc.) |

### 1. How Do Python Data Structures Use Memory?

* Python automatically manages memory, so there are few opportunities to learn about memory.
* Let's examine memory usage.

```python
a = 1
b = a

# id(object): returns the unique identifier (memory address) of an object
print(id(a) == id(b))   # True
```

Variables a and b don't have the same value in different memory address spaces, but rather **share the same memory address**: a and b are two names pointing to the exact same object

* Python variables are 'pointer-like' in that they store memory addresses rather than actual values. More precisely, Python variables are references to objects (similar to pointers).
  * However, unlike pointers, you cannot directly manipulate addresses, so direct pointer operations are not possible.
  * Variables only serve to name objects, while the objects themselves exist in memory.
* **In Python, everything is an object, and variables store references to those objects.**

---

* Mutable Objects vs Immutable Objects

| Object Type               | Examples                              | Characteristics                                               |
| ----------------------- | --------------------------------- | -------------------------------------------------- |
| Mutable     | `list`, `dict`, `set`, etc.          | Changes to the referenced object are reflected in all variables         |
| Immutable | `int`, `float`, `str`, `tuple`, etc. | When a value changes, a new object is created, and the variable references it |

```python
a = 1
b = a

print(id(a))
print(id(b))
```

```txt
140725797079480
140725797079480
```

```python
b = [1,2]
print(id(b))
print(id(b[0]))
print(id(b[1]))

print()

b[0] = 5
print(id(b))
print(id(b[0]))     # Only this value changes
print(id(b[1]))
```

```txt
2430425259456
140725797079480
140725797079512

2430425259456
140725797079608
140725797079512
```

```python
c = [1, 2, 3]
print(id(c))
print(id(c[0]))
print(id(c[1]))
print(id(c[2]))

print()

del c[1]
print(id(c))
print(id(c[0]))
print(id(c[1]))     # Uses the previous id(c[2]) value
```

```txt
2430419924736
140725797079480
140725797079512
140725797079544

2430419924736
140725797079480
140725797079544
```

### 2. Implementing Abstract Data Structures Using Arrays

#### 1) Stack

```python
class Stack():
    def __init__(self):
        self.stack = []

    def isEmpty(self):
        is_empty = False
        if len(self.stack) == 0:
            is_empty = True
        return is_empty

    def push(self, data):
        self.stack.append(data)

    def pop(self):
        pop_object = None
        if self.isEmpty():
            print("Stack is empty")
        else:
            pop_object = self.stack.pop()
        return pop_object

    # Check the last inserted value (the value that will be removed first)
    def top(self):
        top_object = None
        if self.isEmpty():
            print("Stack is empty")
        else:
            top_object = self.stack[-1]
        return top_object

mystack = Stack()
print(mystack.stack)
print(mystack.isEmpty())
```

```txt
[]
True
```

```python
mystack.push(1)
mystack.push(2)
mystack.push(3)

print(mystack.stack)
print(mystack.isEmpty())
```

```txt
[1, 2, 3]
False
```

```python
print(mystack.pop())
print(mystack.stack)
```

```txt
3
[1, 2]
```

```python
print(mystack.top())
```

```txt
2
```

#### 2) Queue

```python
class Queue():
    def __init__(self):
        self.queue = []

    def isEmpty(self):
        is_empty = False
        if len(self.queue) == 0:
            is_empty = True
        return is_empty

    def enqueue(self, data):
        self.queue.append(data)

    def dequeue(self):
        if self.isEmpty():
            print("Queue is empty")
            dequeued = None
        else:
            dequeued = self.queue[0]
            self.queue = self.queue[1:]
        return dequeued

    # Check the first inserted value (the value that will be removed first)
    def peek(self):
        if self.isEmpty():
            print("Queue is empty")
            peeked = None
        else:
            peeked = self.queue[0]
        return peeked

myqueue = Queue()
print(myqueue.queue)
print(myqueue.isEmpty())
```

```txt
[]
True
```

```python
myqueue.enqueue(1)
myqueue.enqueue(2)
myqueue.enqueue(3)

print(myqueue.queue)
print(myqueue.isEmpty())
```

```txt
[1, 2, 3]
False
```

```python
print(myqueue.dequeue())
print(myqueue.queue)
```

```txt
1
[2, 3]
```

```python
print(myqueue.peek())
```

```txt
2
```
#### 3) Tree

```python
class Node():
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree():
    def __init__(self, Node):
        self.root = Node
```

```python
root = Node(10)
root.left = Node(20)
root.right = Node(30)

root.left.left = Node(40)
root.left.right = Node(50)

print(root.data)
print(root.left.data)
print(root.right.data)
print(root.left.left.data)
print(root.left.right.data)
```

```txt
10
20
30
40
50
```

```python
# You can also use the BinaryTree class created above
n1 = Node(10)
n2 = Node(20)
n3 = Node(30)
n4 = Node(40)
n5 = Node(50)       # n1 ~ n5 are all instances of the Node class

tree = BinaryTree(n1)
n1.left = n2
n1.right = n3
n2.left = n4
n2.right = n5

print(tree.root.data)
print(tree.root.left.data)
print(tree.root.right.data)
print(tree.root.left.left.data)
print(tree.root.left.right.data)
```

```txt
10
20
30
40
50
```

##### (1) Preorder Traversal

(It's helpful to review the `1. 1) Recursive Functions` section in the `Algorithm` post before understanding the traversal functions, as they use recursion.)

**Root Node → Left Node → Right Node** order of traversal

```python
def preorder(node):
    if node:
        print(node.data)
        preorder(node.left)
        preorder(node.right)
```

```python
m1 = Node(10)
m2 = Node(20)
m3 = Node(30)
m4 = Node(40)
m5 = Node(50)
m6 = Node(60)
m7 = Node(70)

tree1 = BinaryTree(m1)
m1.left = m2
m1.right = m3
m2.left = m4
m2.right = m5
m4.left = m6
m4.right = m7

preorder(tree1.root)
```

```txt
10
20
40
60
70
50
30
```

##### (2) Inorder Traversal

**Left Node → Root Node → Right Node** order of traversal

```python
def inorder(node):
    if node:
        inorder(node.left)
        print(node.data)
        inorder(node.right)
```

```python
inorder(tree1.root)
```

```txt
60
40
70
20
50
10
30
```

##### (3) Postorder Traversal

**Left Node → Right Node → Root Node** order of traversal

```python
def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        print(node.data)
```

```python
postorder(tree1.root)
```

```txt

```

#### 4) Hash Table

* A function that can be used to map data of arbitrary size to fixed-size data
* Python's `dict` and `set` are representative hash table-based structures

<div align="center">
  <img src="/assets/images/programming/Data Structure and Algorithm_hash_table.svg" width=600 alt="Algorithm_hash_table">  
  <br>
  (Source: <a href="https://simplerize.com/data-structures/hash-table-introduction">Hash Table in Data Structure</a>)
</div>

##### (Reference) Hash: Converting data of arbitrary length into a **fixed-length unique value (hash value)**

>* Representative hash algorithms: SHA-256, MD5, SHA-1
>* One-way: Cannot be reversed to original data (irreversible) $\quad$  _- This is different from encryption_
>* Fast computation speed
>* Collision minimization: Different inputs → different hash values

* Comparing English and Korean encoding

```python
string_en = 'a'
str_encode_en = string_en.encode('utf8')
print('Encoding: ',str_encode_en)

str_decode_en = str_encode_en.decode('utf8')
print('Decoding: ',str_decode_en)
```

```txt
Encoding:  b'a'
Decoding:  a
```

```python
string_ko = '아'
str_encode_ko = string_ko.encode('utf8')    
print('Encoding: ', str_encode_ko)    # hex: represents in hexadecimal

str_decode_kr = str_encode_ko.decode('utf8')
print('Decoding: ', str_decode_kr)
```

```txt
Encoding:  b'\xec\x95\x84'
Decoding:  아
```

```python
# Encoding using Python's hashlib library
import hashlib

name = '철수'
nm_encode = name.encode()
print(nm_encode)

nm_hash = hashlib.sha256(nm_encode)
print(nm_hash)

nm_hash_val = nm_hash.hexdigest()
print(nm_hash_val)

# When we sign up for websites, passwords are converted to hashes using sha256 (a cryptographic hash function) like this ('nm_hash_val' in the example above) and stored
# This is why we used sha256 in AWS practice. Encryption through hashing is absolutely essential
# Some unethical sites store the raw input values ('철수' in the example above). Very vulnerable to security → Don't sign up carelessly!
# When you see in the news 'A certain site is in the spotlight for storing user passwords in plain text' - this is what it means

# Passwords stored as hash values cannot be reversed (irreversibility)
# In other words, even the site owner doesn't know what my password is, and when I enter my password, the site checks if the hash value of what I entered matches the stored password hash value and lets me through if they match
```

```txt
b'\xec\xb2\xa0\xec\x88\x98'
<sha256 _hashlib.HASH object @ 0x00000235E09B3F50>
133a61b3559a6f56b8efe3d2b3fc3b73874a31ac26e4ea32166094ded3d5f1c0
```

* Creating a hash table that maps input values to their hash values

```python
class HashTable():
    def __init__(self):
        self.table = {}

    # Input string value and corresponding hash value
    def put(self, key):
        key_encode = key.encode()
        value = hashlib.sha256(key_encode).hexdigest()
        self.table[key] = value

    # Print entire table
    def printTable(self):
        for key in self.table.keys():
            print(key,": ", self.table[key])

    # Print one value corresponding to the key
    def get(self, key):
        print(self.table[key])

    # Remove a specific value from the list
    def remove(self, key):
        del self.table[key]
```

```python
myhash = HashTable()

myhash.put('철수')
myhash.put('영희')
myhash.put('채원')
myhash.put('소영')
myhash.put('철원')

myhash.get('철수')
myhash.get('영희')
myhash.get('채원')
myhash.get('소영')
myhash.get('철원')
```

```txt
133a61b3559a6f56b8efe3d2b3fc3b73874a31ac26e4ea32166094ded3d5f1c0
2de7bd1bea5aca5d7435eadc5f1ab608007033067a1428cd1fb766185d043723
e82380f937166510e6cd9e1870264ef7a4f0923b775a4fe13a099c05d6a2577b
efdce82c1b00ab877b6b49e3c350b731194376fe1d515925f6e98db2d1f1c63b
2bebf129e109e77f45ab9fd25a9750e27befc7e429dc7f934c4b3ebd382c9903
```

```python
myhash.printTable()
```

```txt
철수 :  133a61b3559a6f56b8efe3d2b3fc3b73874a31ac26e4ea32166094ded3d5f1c0
영희 :  2de7bd1bea5aca5d7435eadc5f1ab608007033067a1428cd1fb766185d043723
채원 :  e82380f937166510e6cd9e1870264ef7a4f0923b775a4fe13a099c05d6a2577b
소영 :  efdce82c1b00ab877b6b49e3c350b731194376fe1d515925f6e98db2d1f1c63b
철원 :  2bebf129e109e77f45ab9fd25a9750e27befc7e429dc7f934c4b3ebd382c9903
```

```python
myhash.remove('철수')
myhash.printTable()
```

```txt
영희 :  2de7bd1bea5aca5d7435eadc5f1ab608007033067a1428cd1fb766185d043723
채원 :  e82380f937166510e6cd9e1870264ef7a4f0923b775a4fe13a099c05d6a2577b
소영 :  efdce82c1b00ab877b6b49e3c350b731194376fe1d515925f6e98db2d1f1c63b
철원 :  2bebf129e109e77f45ab9fd25a9750e27befc7e429dc7f934c4b3ebd382c9903
```

#### 5) Graph

<img src="/assets/images/programming/Data Structure and Algorithm_graph.jpg" width=600 alt="Data Structure and Algorithm_graph">

* Note) Trees are also included in graphs: The key of each dictionary represents the parent Node of the tree, and the values represent child Nodes

```python
# Implemented with hash table: Map `nodes` to [paths the node can take]
class HashMap():
    def __init__(self):
        self.table = {}

    def put(self, key, value):
        self.table[key] = value

    def printTable(self):
        for key in self.table.keys():
            print(key, ":", self.table[key])

    def get(self, key):
        print(self.table[key])

    def remove(self, key):
        del self.table[key]
```

```python
graph = HashMap()
graph.put(1, [2, 3, 4])
graph.put(2, [1, 4, 5])
graph.put(3, [1, 4])
graph.put(4, [1, 2, 3, 5])
graph.put(5, [2, 4])

graph.printTable()
```

```txt
1 : [2, 3, 4]
2 : [1, 4, 5]
3 : [1, 4]
4 : [1, 2, 3, 5]
5 : [2, 4]
```

```python
# Writing directly in dictionary format
graph = {
    1: [2, 3, 4],
    2: [1, 4, 5],
    3: [1, 4],
    4: [1, 2, 3, 5],
    5: [2, 4]
}

graph
```

```txt
{1: [2, 3, 4], 2: [1, 4, 5], 3: [1, 4], 4: [1, 2, 3, 5], 5: [2, 4]}
```

* What if you want to reflect edge weights?
  * You can expand each value like this: `starting Node: [(connected Node1, weight1), (connected Node2, weight2), ...]`

Example)

<img src="/assets/images/programming/Data Structure and Algorithm_graph_weighted.jpg" width=600 alt="Data Structure and Algorithm_graph_weighted" style="display: block; margin: 0;">

```python
graph_weighted = {
    1: [(2, 10), (3, 30)],
    2: [(4, 40), (5, 60)],
    3: [(4, 20)],
    4: [(1, 50), (5, 70)],
    5: []
}

graph_weighted
```

```txt
{1: [(2, 10), (3, 30)],
 2: [(4, 40), (5, 60)],
 3: [(4, 20)],
 4: [(1, 50), (5, 70)],
 5: []}
```
