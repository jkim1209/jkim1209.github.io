---
layout: post
title: "파이썬 자료구조"
date: 2025-06-26
categories: [Programming, Data Structure]
tags: [python, data structure, stack, queue, tree, hash-table, graph, algorithms, adt]
math: true
---

## [자료구조] Data Structure

### 0. 파이썬 자료구조 종류

| 용어                                                                       | 설명                                |
| -------------------------------------------------------------------------- | ----------------------------------- |
| `list`, `tuple`, `set`, `frozenset`, `dict`, `str`                         | 파이썬 언어 내장 자료구조           |
| 스택(stack), 큐(queue), 해시 테이블(hash table), 트리(tree), 그래프(graph) | 추상 자료구조 (ADT), 직접 구현 필요 |

* 파이썬 언어 내장 자료구조

| 내장 자료구조 | 설명                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| `list`        | 순서 있고 변경 가능한 시퀀스 (동적 배열: 필요에 따라 크기를 자동으로 조절) |
| `tuple`       | 순서 있고 변경 불가능한 시퀀스                                             |
| `set`         | 중복 없는 변경 가능한 집합                                                 |
| `frozenset`   | 변경 불가능한 집합                                                         |
| `dict`        | 키-값 쌍의 해시 기반 매핑                                                  |
| `str`         | 변경 불가능한 문자열 시퀀스                                                |

* 추상 자료구조

| 자료구조 이름           | 선형 / 비선형 | 간단한 설명                                              |
| ----------------------- | ------------- | -------------------------------------------------------- |
| 스택 (Stack)            | 선형 구조     | 후입선출(LIFO) 방식으로 데이터를 저장하고 꺼냄           |
| 큐 (Queue)              | 선형 구조     | 선입선출(FIFO) 방식으로 데이터를 저장하고 꺼냄           |
| 트리 (Tree)             | 비선형 구조   | 계층적인 부모-자식 관계를 가지는 노드 구조               |
| 해시테이블 (Hash Table) | 비선형 구조   | 키를 해시함수로 변환해 빠르게 데이터에 접근하는 구조     |
| 그래프 (Graph)          | 비선형 구조   | 노드(정점)와 그 사이의 연결(간선)로 구성된 일반화된 구조 |

_Note_ : **자료형**(Data Type) vs **자료구조**(Data Structure)  

| 개념                          | 의미                                                               |
| ----------------------------- | ------------------------------------------------------------------ |
| **자료형** (Data Type)        | 값의 **종류와 크기**를 정의함 (예: `int`, `float`, `str`)          |
| **자료구조** (Data Structure) | 데이터를 **저장하고 조직화**하는 구조 (예: `list`, `set`, 트리 등) |

### 1. 파이썬 자료구조는 메모리를 어떻게 쓸까?

* 파이썬에서는 메모리를 자동으로 관리를 해 주어서 메모리에 대해 알 기회가 거의 없다.
* 메모리에 대해 살펴본다.

```python
a = 1
b = a

# id(객체): 객체의 고유 식별자(메모리 주소)를 반환
print(id(a) == id(b))   # True
```

변수 a와 b는 동일한 값만 가지고 서로 다른 메모리 주소 공간을 갖는게 아니라 **서로 같은 메모리 주소**를 갖는다 : a와 b는 완벽히 동일한 객체를 가리키는 두 개의 이름

* 실제 값이 아닌 메모리 주소를 저장한다는 점에서 파이썬의 변수는 '포인터 같다'. 정확히 말하면 파이썬의 변수는 객체에 대한 참조(포인터 비슷한 것)임.  
  * 하지만 포인터처럼 주소를 직접 다루지는 않아 직접적인 포인터 연산은 불가.  
  * 변수는 객체에 이름을 붙이는 역할을 할 뿐이며, 객체 자체는 메모리에 존재함.  
* **파이썬에서는 모든 것이 객체이며, 변수는 그 객체의 참조를 저장한다.**

---

* 변경 가능한 객체 vs 변경 불가능한 객체  

| 객체 타입               | 예시                              | 특징                                               |
| ----------------------- | --------------------------------- | -------------------------------------------------- |
| 변경 가능 (mutable)     | `list`, `dict`, `set` 등          | 참조된 객체 내용이 바뀌면 모든 변수에 반영         |
| 변경 불가능 (immutable) | `int`, `float`, `str`, `tuple` 등 | 값이 바뀌면 새 객체가 만들어지고, 변수는 그걸 참조 |

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
print(id(b[0]))     # 이 값만 바뀐다.
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
print(id(c[1]))     # 기존의 id(c[2])값을 사용하게 된다.
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

### 2. 배열을 이용한 추상 자료구조 구현

#### 1) 스택(stack)

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
    
    # 마지막에 들어간 값 확인 (다음에 제일 먼저 꺼내질 값)
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

#### 2) 큐(queue)

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
    
    # 처음 들어간 값 확인 (다음에 제일 먼저 꺼내질 값)
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

#### 3) 트리(tree)

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
# 위의 BinaryTree 클래스를 만들어 이용할 수도 있다.
n1 = Node(10)
n2 = Node(20)
n3 = Node(30)
n4 = Node(40)
n5 = Node(50)       # n1 ~ n5 모두 Node 클래스가 되었다.

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

##### (1) 전위 순회  

(순회함수를 만들 때 재귀함수가 쓰이니 `Algorithm` 포스트의 `1. 1) 재귀함수` 부분을 먼저 보고 오면 이해가 편하다.)

**루트 노드 → 왼쪽 노드 → 오른쪽 노드** 순서로 탐색

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

##### (2) 중위 순회

**왼쪽 노드 → 루트 노드 → 오른쪽 노드** 순서로 탐색

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

##### (3) 후위 순회

**왼쪽 노드 → 오른쪽 노드 → 루트 노드** 순서로 탐색

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

#### 4) 해시테이블(hash table)

* 임의 크기 데이터를 고정된 길이의 데이터로 매핑하는데 사용할 수 있는 함수  
* 파이썬의 `dict`, `set`이 대표적인 해시테이블 기반 구조  

<div align="center">
  <img src="/assets/images/programming/Data Structure and Algorithm_hash_table.svg" width=600 alt="Algorithm_hash_table">  
  <br>
  (출처: <a href="https://simplerize.com/data-structures/hash-table-introduction">Hash Table in Data Structure</a>)
</div>

##### (참고) 해시: 임의의 길이의 데이터를 **고정된 길이의 고유한 값(해시값)**으로 변환하는 것

>* 대표적인 해시 알고리즘: SHA-256, MD5, SHA-1  
>* 일방향성: 원래 데이터로 되돌릴 수 없음 (복호화 불가능) $\quad$  _- 이러한 점에서 암호화(Encryption)와는 다름_  
>* 빠른 계산 속도  
>* 충돌 최소화: 서로 다른 입력 → 다른 해시값  

* 영어와 한글 인코딩 비교  

```python
string_en = 'a'
str_encode_en = string_en.encode('utf8')
print('인코딩: ',str_encode_en)

str_decode_en = str_encode_en.decode('utf8')
print('디코딩: ',str_decode_en)
```

```txt
인코딩:  b'a'
디코딩:  a
```

```python
string_ko = '아'
str_encode_ko = string_ko.encode('utf8')    
print('인코딩: ', str_encode_ko)    # 헥스(hex): 16진법으로 나타냄

str_decode_kr = str_encode_ko.decode('utf8')
print('디코딩: ', str_decode_kr)
```

```txt
인코딩:  b'\xec\x95\x84'
디코딩:  아
```

```python
# 파이썬 라이브러리 hashlib을 사용하여 인코딩해보기
import hashlib

name = '철수'
nm_encode = name.encode()
print(nm_encode)

nm_hash = hashlib.sha256(nm_encode)
print(nm_hash)

nm_hash_val = nm_hash.hexdigest()
print(nm_hash_val)

# 우리가 사이트에 가입할 때 쓰는 암호는 이와 같이 sha256(암호학적 해시 함수의 하나)으로 해시로 바꾸어져(위 예시에서 'nm_hash_val') 저장된다.
# aws실습 때 sha256을 썼던 것도 이런 이유임. 해시를 통한 암호화는 반드시 필수.
# 가끔 양심없이 입력한 값(위 예시에서 '철수') 그대로 저장되는 사이트들이 있음. 보안에 매우 취약 → 함부로 가입하고 다니지 말 것!
# 뉴스에서 '모사이트에서 사용자의 암호를 평문으로 저장해서 화제가 되고 있습니다' 가 이런 뜻이다.

# 이렇게 해시값으로 바뀌어 저장된 암호는 다시 돌리지 못함(비가역성)
# 즉, 사이트 주인도 나의 암호가 뭔지 모르고, 사이트에서는 내가 비밀번호를 입력하면 그 해시값이 저장된 나의 암호 해시값과 일치하면 통과시켜주는 것이다. 
```

```txt
b'\xec\xb2\xa0\xec\x88\x98'
<sha256 _hashlib.HASH object @ 0x00000235E09B3F50>
133a61b3559a6f56b8efe3d2b3fc3b73874a31ac26e4ea32166094ded3d5f1c0
```

* 입력값과 입력값의 해시값을 매핑하는 해시 테이블 생성하기  

```python
class HashTable():
    def __init__(self):
        self.table = {}
    
    # 문자값과 대응되는 해시값 입력
    def put(self, key):
        key_encode = key.encode()
        value = hashlib.sha256(key_encode).hexdigest()
        self.table[key] = value

    # 전체 테이블 출력
    def printTable(self):
        for key in self.table.keys():
            print(key,": ", self.table[key])
    
    # key에 해당하는 값 하나를 출력
    def get(self, key):
        print(self.table[key])

    # 목록에서 특정 값 제거
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

#### 5) 그래프(graph)

<img src="/assets/images/programming/Data Structure and Algorithm_graph.jpg" width=600 alt="Data Structure and Algorithm_graph"> 

* 참고) 트리도 그래프 안에 포함된다: 각 딕셔너리의 key가 트리의 부모 Node, value들은 자식 Node로 표현

```python
# 해시 테이블로 구현: `노드`와 [노드가 갈 수 있는 경로]를 매핑
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
# 직접 딕셔너리 형태로 쓰기
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

* 간선의 가중치도 반영하고 싶다면?
  * 각 값을 확장시켜 `시작 Node: [(연결된 Node1, 가중치1), (연결된 Node2, 가중치2), ...]` 이런 식으로 쓸 수 있다.

예시)

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
