---
layout: post
title: "알고리즘"
date: 2025-06-27
categories: [Programming, Algorithm]
tags: [algorithms, search-algorithms, sorting-algorithms, recursion, bfs, dfs, binary-search, merge-sort, quick-sort, memoization, dynamic-programming, python]
math: true
---

## [알고리즘] Algorithm

### 0. 알고리즘이란?

* 자료구조가 데이터를 조직화하고 저장하는 방식에 대한 것이었다면, 알고리즘은 문제를 해결하기 위한 단계적 절차이다.
* 알고리즘은 자료구조를 이용하여 특정 문제를 해결하거나 특정 작업을 수행하는 방법이다.
* 크게 탐색과 정렬로 나누어진다.  

### 1. 탐색 알고리즘

#### 1) 재귀 함수

* 함수 내에서 해당 함수를 다시 사용하는 형태
* 재귀함수는 함수를 다시 자기 자신을 호출하면서 호출된 함수가 스택(stack)에 쌓이고, 나중에 호출된 함수부터 차례로 되돌아가며(pop) 실행을 마침
* 이건 정확히 스택 구조, 즉 **후입선출(LIFO: Last-In First-Out)** 의 동작 방식과 같음  

>* 재귀함수는 자체로 탐색 알고리즘은 아니지만, 자료구조나 알고리즘을 표현하기 위한 기법(기술)으로 반복문처럼 문제를 분할하고 해결하는 방식으로 자주 사용됨.
>* 특히 반복문처럼 문제를 분할하고 해결하는 방식으로 자주 사용됨.
>   * 예: 팩토리얼, 피보나치 수열, 하노이 탑 등

피보나치 수열 예시)

```python
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n > 1:
        return fib(n-1) + fib(n-2)
    else:
        class fibError(Exception):
            pass
        raise fibError("n must be positive!")
```

<div align="center">
  <img src='/assets/images/programming/Data Structure and Algorithm_recursive_function.png' width="700" alt="Data Structure and Algorithm_recursive_function">
  (출처: <a href="https://www.eecs.yorku.ca/course_archive/2016-17/F/2030/labs/lab7/lab7.html" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">Emulating recursion using a stack</a>)
</div>

```python
def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)
# 함수 구조를 보면 n = n에서 시작하여 n = 0이 될 때까지 계속 factorial 함수가 반복 순환하면서(recursively) 돌아간다.

factorial(5)
```

```txt
120
```

#### 2) 너비 우선 탐색(BFS, Breadth First Search)

* 트리 구조에서 가로방향으로 탐색  
* BFS는 **큐(queue) 자료 구조** 사용 → **선입선출(FIFO)**  
  * 너비우선으로 자료를 탐색하며 넣은 뒤 추출 (FIFO)  
  * 어떤 자료를 추출할 때 그 자료의 자식 노드들을 추가  
  * 이를 계속 반복  

**예시)**  

<img src="/assets/images/programming/Data Structure and Algorithm_bfs.jpg" width="600" alt="Data Structure and Algorithm_bfs" style="display: block; margin: 0;">

> [1] 1을 큐에추가한 뒤 추출, 그 자식 노드들을 append하면 순서대로 8, 5, 2 가 들어감  
> [2] 1 다음으로 들어간 8을 추출, 그 자식 노드들을 append하면 순서대로 6, 4, 3 이 들어감  
> $\quad$ 하지만 이 값들은 append되었으므로 여전히 5, 2 는 6, 4, 3 앞에 남아있음  
> [3] 8 다음으로 들어간 5를 추출, 그 자식 노드들은 없으므로 pass  
> $\,$ ... 이와 같이 반복한다: FIFO방식으로 넓이 우선 탐색 가능  

```python
# 우선 위 트리를 그래프 형태로 만들기
graph = {
    1: [8, 5, 2],
    8: [6, 4, 3],
    5: [],
    2: [9],
    6: [10, 7],
    4: [],
    3: [],
    9: [],
    10: [],
    7: []
}

# 방문했는지 여부를 딕셔너리 형태로 만들기 (처음엔 모두 거짓)
visited = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
    7: False,
    8: False,
    9: False,
    10: False
}
```

```python
def bfs(graph, visited):
    queue = []
    root = list(graph.keys())[0]
    queue.append(root)
    visited[root] = True   # 루트노드는 처음에 방문하게 되므로 True로 선언
    while queue:
        vertex = queue.pop(0)    # queue에 들어간 값을 선입선출 방식으로 빼줌
        print(vertex, end=' ')
        for neighbor in graph[vertex]:
            if not visited[neighbor]:       # 아직 방문하지 않은 곳이라면,
                queue.append(neighbor)
                visited[neighbor] = True     # 방문한 노드는 True로 설정

# 참고: for문 vs while문
# for문은 처음부터 loop을 몇 번 돌릴지 정해서 돌림
# while문은 조건이 만족할때까지 loop를 돌림 (loop가 몇 번 돌아갈지 설정 안 함)
```

```python
bfs(graph, visited)
```

```txt
1 8 5 2 6 4 3 9 10 7
```

#### 3) 깊이 우선 탐색(DFS, Depth First Search)

* 트리 구조에서 세로방향으로 탐색  
* BFS는 **스택(stack) 자료 구조** 사용 → **후입선출(LIFO)**  
  * 즉, 깊이우선으로 자료를 **모두** 탐색, 자식노드가 더 이상 없으면 멈춤 $\quad$ _- 재귀함수 이용_  
  * 한 방향(보통 왼쪽부터) 가능한 만큼 끝까지 내려간 뒤, 더 이상 갈 곳이 없으면 되돌아(backtrack) 하며 다른 방향을 탐색함.  
  * 탐색이 끝났다면, 마지막으로 탐색된 값부터 append  
  * 재귀함수 이용  

**예시)**  

<img src="/assets/images/programming/Data Structure and Algorithm_dfs.jpg" width="600" alt="Data Structure and Algorithm_dfs" style="display: block; margin: 0;">

> [1] 재귀함수를 통해 자식노드가 없을 때까지 계속해서 왼쪽부터 수직아래로 내려가며 자료를 탐색  
> [2] 탐색이 종료되면 마지막으로 탐색된 값부터 하나씩 append : 9 → 2 → 5 → 3 → 4 → 7 → 10 → 6 → 8 → 1  
> [3] 마지막에 들어간 값부터 하나씩 추출해내면 LIFO방식으로 깊이 우선 탐색 가능 : 1 8 6 10 7 4 3 5 2 9  

```python
# 우선 위 트리를 그래프 형태로 만들기
graph = {
    1: [8, 5, 2],
    8: [6, 4, 3],
    5: [],
    2: [9],
    6: [10, 7],
    4: [],
    3: [],
    9: [],
    10: [],
    7: []
}

# 방문했는지 여부를 딕셔너리 형태로 만들기 (처음엔 모두 거짓)
visited = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
    7: False,
    8: False,
    9: False,
    10: False
}
```

```python
def dfs(graph, vertex, visitied):
    visited[vertex] = True
    print(vertex, end = ' ')
    for neighbor in graph[vertex]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited)
```

```python
dfs(graph, 1, visited)
```

```txt
1 8 6 10 7 4 3 5 2 9
```

#### 4) 선형 탐색(linear search)

* 순차 탐색(sequential search)이라고도 불림  
* `for`문 돌리는 것이 바로 선형 탐색  
* 장점: 구형하기 쉽고, 정렬되지 않은 상태에서도 사용 가능  
* 단점: 검색 대상 리스트가 길어지면 비효율적 - (최대) 시간 복잡도: $O(n)$  

<div align="center">
  <img src="/assets/images/programming/Data Structure and Algorithm_linear_search.png" width="600" alt="Data Structure and Algorithm_linear_search" style="display: block; margin: 0;">
  <a href="https://www.geeksforgeeks.org/dsa/what-is-linear-search/">What is Linear Search?</a>
</div>

```python
# for문을 바로 사용하는 것 외에 선형 탐색을 함수로 만들어보기
def linearsearch(item_list, target):
    n = len(item_list)
    for i in range(0, n):
        item = item_list[i]
        if item == target:
            return i
    return -1
```

```python
item_list = [31, 15, 12, 95, 25, 74, 22, 84, 26, 67]
target1 = 25
target2 = 11

idx1 = linearsearch(item_list, target1)
idx2 = linearsearch(item_list, target2)

print(idx1)
print(idx2)
```

```txt
4
-1
```

#### 5) 이진 탐색(binary search) - 반복문 사용

* 장점: 선형 탐색에 비해 빠른 속도 - 시간 복잡도: $O(log{n})$  
* 단점: 미리 오름차순 정렬되어 있어야 사용 가능  

<div align="center">
  <img src="/assets/images/programming/Data Structure and Algorithm_binary_search.png" width="600" alt="Data Structure and Algorithm_binary_search" style="display: block; margin: 0;">
  <a href="https://dolly-desir.medium.com/algorithms-binary-search-2656c7eb5049">Algorithms: Binary Search</a>
</div>

```python
def binarySearchLoop(item_list, target):
    start = 0
    end = len(item_list) - 1

    while start <= end:
        mid = (start + end) // 2            # // : 몫, 즉 소수점 이하를 버리고 정수만 남김

        if item_list[mid] < target:
            start = mid + 1                 # mid 뒤쪽에 있으니 start 인덱스를 mid + 1 로 설정
        elif item_list[mid] > target:
            end = mid - 1                   # mid 앞쪽에 있으니 end 인덱스를 mid - 1 로 설정
        else:
            return mid
    return -1
```

```python
item_list = [1, 7, 11, 16, 23, 28, 31, 35, 41, 50]
target = 16

res = binarySearchLoop(item_list, target)
print(res)
```

```txt
3
```

### 2. 정렬 알고리즘

#### 1) 버블 정렬(bubble sort)

* 거품 정렬  
* 앞에서부터 두 인접한 원소를 검사한 후 정렬하는 방법  
* 시간 복잡도: $O(n^2)$  
* 구현이 간단함  

```python
def bubble_sort(item_list):
    n = len(item_list)
    for i in range(1, n):
        for j in range(0, n-1):
            if item_list[j] > item_list[j+1]:                                   # 뒤의 값이 더 크다면
                item_list[j], item_list[j+1] = item_list[j+1], item_list[j]     # 순서 바꿔주기
    return item_list
```

```python
item_list = [2, 5, 3, 1, 9]
bubble_sort(item_list)
```

```txt
[1, 2, 3, 5, 9]
```

#### 2) 병합 정렬(merge sort)

* 안정적인 알고리즘  
* 분할 정복 알고리즘: 간단한 문제가 될 때까지 문제를 재귀적으로 분할한 다음, 최하위 문제들을 해결함으로써 원래 문제의 답을 찾아냄  
* 상용 라이브러리에 많이 쓰임  
* 시간 복잡도: $O(nlog(n))$  

<img src="/assets/images/programming/Data Structure and Algorithm_merge_sort.jpg" width="1200" alt="Data Structure and Algorithm_merge_sort" style="display: block; margin: 0;">

> 따라서 병합 정렬을 구현하려면 **분할하는 함수**와 **병합하는 함수** 두 개가 필요함  

```python
# 병합하는 함수
def merge(left, right):
    print("========== merge 함수 ==========")   
    print("left: ", left, "right: ", right)
    sorted_list = []                            # 다음 병합에 쓰일 자료들을 모아놓은 list

    while len(left) > 0 or len(right) > 0:      # left나 right에 값이 있는 동안 계속 진행
        if len(left) > 0 and len(right) > 0:    # left와 right 양쪽에 값이 있는 경우
            if left[0] <= right[0]:             
                sorted_list.append(left[0])     # left의 첫번째 값과 right의 첫번째 값을 비교하여 작은 값을 sorted_list에 넣기
                left = left[1:]                 # 넣어준 값은 제외    
                                                    # 파이썬에서 `인덱싱`(e.g. lst[1])은 범위 초과에 오류가 발생하지만 
                                                    # `슬라이싱`(e.g. lst[1:])은 범위 초과에도 오류가 없이 값이 없는 부분은 비어 있는 채로 반환한다!!
            else:
                sorted_list.append(right[0])    # left의 첫번째 값과 right의 첫번째 값을 비교하여 작은 값을 sorted_list에 넣기
                right = right[1:]               # 넣어준 값은 제외

        elif len(left) > 0:                     # left에만 값이 있는 경우
            sorted_list.append(left[0])         # 해당 값을 sorted_list에 넣기
            left = left[1:]

        elif len(right) > 0:                    # right에만 값이 있는 경우
            sorted_list.append(right[0])        # 해당 값을 sorted_list에 넣기
            right = right[1:]
            
    print("sorted_list: ", sorted_list)
    print("================================")
    return sorted_list


# 분할하는 함수 + 병합하는 함수
def merge_sort(item_list):
    print("======== merge_sort 함수 ========")
    if len(item_list) <= 1:
        return item_list
    
    mid = len(item_list) // 2
    left = item_list[:mid]
    right = item_list[mid:]
    print("left: ", left)
    print("right: ", right)

    left_rec = merge_sort(left)         # mid기준으로 왼쪽 부분에 대해서도 더 이상 쪼갤 수 없을 때까지 (len(item_list) <=1 일 때까지) 계속 똑같은 작업을 수행
    right_rec = merge_sort(right)       # mid기준으로 오른쪽 부분에 대해서도 더 이상 쪼갤 수 없을 때까지 (len(item_list) <=1 일 때까지) 계속 똑같은 작업을 수행

    return merge(left_rec, right_rec)   # 병합
```

```python
item_list = [33, 25, 42, 1, 8, 51, 12]
merge_sort(item_list)
```

```txt
======== merge_sort 함수 ========
left:  [33, 25, 42]
right:  [1, 8, 51, 12]
======== merge_sort 함수 ========
left:  [33]
right:  [25, 42]
======== merge_sort 함수 ========
======== merge_sort 함수 ========
left:  [25]
right:  [42]
======== merge_sort 함수 ========
======== merge_sort 함수 ========
========== merge 함수 ==========
left:  [25] right:  [42]
sorted_list:  [25, 42]
================================
========== merge 함수 ==========
left:  [33] right:  [25, 42]
sorted_list:  [25, 33, 42]
================================
======== merge_sort 함수 ========
left:  [1, 8]
right:  [51, 12]
======== merge_sort 함수 ========
left:  [1]
right:  [8]
======== merge_sort 함수 ========
======== merge_sort 함수 ========
========== merge 함수 ==========
left:  [1] right:  [8]
sorted_list:  [1, 8]
================================
======== merge_sort 함수 ========
left:  [51]
right:  [12]
======== merge_sort 함수 ========
======== merge_sort 함수 ========
========== merge 함수 ==========
left:  [51] right:  [12]
sorted_list:  [12, 51]
================================
========== merge 함수 ==========
left:  [1, 8] right:  [12, 51]
sorted_list:  [1, 8, 12, 51]
================================
========== merge 함수 ==========
left:  [25, 33, 42] right:  [1, 8, 12, 51]
sorted_list:  [1, 8, 12, 25, 33, 42, 51]
================================
```

```txt
[1, 8, 12, 25, 33, 42, 51]
```

#### 3) 퀵 정렬(quick sort)

* 데이터의 초기 정렬 상태에 영향을 받음
* 피벗의 선택이 중요
* 평균 시간 복잡도: $O(nlog(n))$
* 최악 시간 복잡도: $O(n^2)$

<img src="/assets/images/programming/Data Structure and Algorithm_quick_sort1.jpg" width="600" alt="Data Structure and Algorithm_quick_sort1" style="display: block; margin: 0;">  

> [1] pivot을 0번째 값으로 지정했다고 하자.  
> [2] 이 pivot을 기준으로 list를 탐색하며 pivot보다 [작은 값들], [같은 값들], [큰 값들]로 모아서 나눠준다.  

<img src="/assets/images/programming/Data Structure and Algorithm_quick_sort2.jpg" width="600" alt="Data Structure and Algorithm_quick_sort2" style="display: block; margin: 0;">  

<img src="/assets/images/programming/Data Structure and Algorithm_quick_sort3.jpg" width="600" alt="Data Structure and Algorithm_quick_sort3" style="display: block; margin: 0;">  

> [3] 나눠진 [작은 값들], [큰 값들]에 대해서 다시 quick_sort를 해준다.  
> $\\quad$ _- 재귀함수 이용: 따라서 이 때에도 pivot은 계속 위에서 정한 pivot과 같이 각 값들의 0번째 값_  
> [4] 이를 나눠진 값들의 길이가 모두 1 이하일 때까지 반복한다.  

<img src="/assets/images/programming/Data Structure and Algorithm_quick_sort4.jpg" width="600" alt="Data Structure and Algorithm_quick_sort4" style="display: block; margin: 0;">  

> [5] 최종적으로 나눠진 값들을 그대로 이어 붙여주면 된다.

```python
def quick_sort(item_list):
    n = len(item_list)
    if n <= 1:
        return item_list
    
    else:
        pivot = item_list[0]
        greater_list = []
        eq_list = []
        less_list = []
        for element in item_list:
            if element > pivot:
                greater_list.append(element)
            elif element < pivot:
                less_list.append(element)
            else:
                eq_list.append(element)
        return quick_sort(less_list) + eq_list + quick_sort(greater_list)   # 재귀 & 나눠진 값들을 그대로 이어 붙여줌
```

```python
lst = [2, 5, 1, 9, 2, 3, 2, 1]
quick_sort(lst)
```

```txt
[1, 1, 2, 2, 2, 3, 5, 9]
```

#### 4) 메모이제이션(memoization)

* 중복 계산을 피하기 위해 이미 계산한 값을 저장해 놓음  
* 중복 계산시 이미 저장되어 있는 값 사용  

재귀함수 vs 메모이제이션

| 개념             | 설명                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------- |
| **재귀함수**     | 함수가 자기 자신을 호출하는 방식 (예: 팩토리얼, 피보나치 등)                          |
| **메모이제이션** | 이미 계산한 값을 저장해두고 동일한 입력에 대해 다시 계산하지 않도록 하는 최적화 기법 |

재귀는 계산을 반복하는 구조.  
메모이제이션은 중복 계산을 피하기 위한 캐싱.  
→ 따라서 둘을 함께 쓰면 효율이 급격히 좋아짐.

예시)

* 가령 피보나치 수열의 6번째 값을 찾으려면 4,5번째 값의 합을 찾아야 하는데,  
* 5번째 값을 찾으려면 3,4번째 값을 알아야 함  
* 4번째 값을 저장해두었다면 바로 꺼내 쓰면 되겠지만(메모이제이션), 저장하지 않은 경우(메모이제이션X)는 중복해서 계산을 하게 된다.  

```python
# 피보나치 수열
# 재귀만 사용 (중복호출 많음)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)


# 메모이제이션 적용
res = {}    # 이미 계산한 값을 딕셔너리로 저장
def fib_memoization(n):
    print(f"실행! → n = {n}")
    if n in res:                # 이미 저장된 게 있으면 불러오라는 코드
        return res[n]
    if n <= 1:
        return n
    res[n] = fib_memoization(n-1) + fib_memoization(n-2)    # 이전 두 값을 더해서 res에 저장
    print(res)
    return res[n]
```

```python
# 실행시간 비교
import time

# 재귀만 사용
start_time1 = time.time()  
fib_recursive_res = fib_recursive(20)
end_time1 = time.time() 

# 메모이제이션 적용
start_time2 = time.time()  
fib_memoization_res = fib_memoization(20)
end_time2 = time.time() 

print()
print("재귀만 사용했을 때 걸린 시간:", end_time1 - start_time1, "초")
print("메모이제이션 적용했을 때 걸린 시간:", end_time2 - start_time2, "초")
```

```txt
실행! → n = 20
실행! → n = 19
실행! → n = 18
실행! → n = 17
실행! → n = 16
실행! → n = 15
실행! → n = 14
실행! → n = 13
실행! → n = 12
실행! → n = 11
실행! → n = 10
실행! → n = 9
실행! → n = 8
실행! → n = 7
실행! → n = 6
실행! → n = 5
실행! → n = 4
실행! → n = 3
실행! → n = 2
실행! → n = 1
실행! → n = 0
{2: 1}
실행! → n = 1
{2: 1, 3: 2}
실행! → n = 2
{2: 1, 3: 2, 4: 3}
실행! → n = 3
{2: 1, 3: 2, 4: 3, 5: 5}
실행! → n = 4
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8}
실행! → n = 5
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13}
실행! → n = 6
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21}
실행! → n = 7
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34}
실행! → n = 8
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55}
실행! → n = 9
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89}
실행! → n = 10
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144}
실행! → n = 11
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233}
실행! → n = 12
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377}
실행! → n = 13
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610}
실행! → n = 14
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987}
실행! → n = 15
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597}
실행! → n = 16
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584}
실행! → n = 17
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181}
실행! → n = 18
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765}

재귀만 사용했을 때 걸린 시간: 0.0 초
메모이제이션 적용했을 때 걸린 시간: 0.00099945068359375 초
```

```arduino
fib(20)
 └── fib(19)
      └── fib(18)
           ...
                └── fib(2)
                     └── fib(1) → 1
                     └── fib(0) → 0
                     → res[2] = 1     ← 여기서부터 올라가며 res 저장 + print
                → res[3] = 2
           ...
     → res[20] = 6765
```

```python
# 30으로 올려 추가적으로 실행해보면?
# 재귀만 사용
start_time1 = time.time()  
fib_recursive_res = fib_recursive(30)
end_time1 = time.time() 

# 메모이제이션 적용
start_time2 = time.time()  
fib_memoization_res = fib_memoization(30)
end_time2 = time.time() 

print()
print("재귀만 사용했을 때 걸린 시간:", end_time1 - start_time1, "초")
print("메모이제이션 적용했을 때 걸린 시간:", end_time2 - start_time2, "초")
```

```txt
실행! → n = 30
실행! → n = 29
실행! → n = 28
실행! → n = 27
실행! → n = 26
실행! → n = 25
실행! → n = 24
실행! → n = 23
실행! → n = 22
실행! → n = 21
실행! → n = 20
실행! → n = 19
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946}
실행! → n = 20
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711}
실행! → n = 21
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657}
실행! → n = 22
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368}
실행! → n = 23
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025}
실행! → n = 24
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025, 26: 121393}
실행! → n = 25
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025, 26: 121393, 27: 196418}
실행! → n = 26
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025, 26: 121393, 27: 196418, 28: 317811}
실행! → n = 27
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025, 26: 121393, 27: 196418, 28: 317811, 29: 514229}
실행! → n = 28
{2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181, 20: 6765, 21: 10946, 22: 17711, 23: 28657, 24: 46368, 25: 75025, 26: 121393, 27: 196418, 28: 317811, 29: 514229, 30: 832040}

재귀만 사용했을 때 걸린 시간: 0.06900453567504883 초
메모이제이션 적용했을 때 걸린 시간: 0.0 초
```

>* 메모이제이션을 적용한 경우 이미 피보나치 수열 20번째 값까지는 저장되어 있으므로 21부터 계산됨을 확인할 수 있다.

#### 5) 다이나믹 프로그래밍(dynamic programming)

* 복잡한 문제를 작은 문제로 나눠서 푸는 기법 (분할 정복)
* 이전에 계산한 결과를 저장한 후 재활용 (메모이제이션과 비슷)

|                | 다이나믹 프로그래밍 | 메모이제이션 |
| :------------: | :-----------------: | :----------: |
|   접근 방식    |      Bottom-up      |   Top-down   |
|   처리 방식    |       반복문        |     재귀     |
| 중간 결과 저장 |          O          |      O       |

```python
# 피보나치 수열
def fibonacci_dp(n):
    if n <= 1:
        return n
    
    dp = [0, 1]     # 초기값 F(0) = 1, F(1) = 1
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n], dp    # 피보나치 수열의 n번째 수, n번째 수까지 피보나치 수열
```

```python
res, dp = fibonacci_dp(20)
print(res)
print(dp)
```

```txt
6765
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
```

```python
# 동전으로 특정 금액을 만들 수 있는 경우의 수 찾기
def count_coin(n):                  # n원을 만들 수 잇는 경우의 수를 구하는 함수
    coins = [10, 50, 100, 500]      # 사용할 수 있는 동전의 종류
    res = [0] * (n+1)               # res[i]: i 원을 만드는 경우의 수. res[0]부터 res[n]까지 총 n+1개의 원소가 필요하므로 n+1 크기로 만듦. 초기에는 모두 0으로 설정
    res[0] = 1                      # 0원 만드는 방법은 1가지 (돈 안씀)

    for coin  in coins:             # 각 동전(10, 50, 100, 500)을 차례로 사용하면서 경우의 수를 누적
                                    # "작은 단위 동전부터 하나씩 누적해서 조합을 완성해나간다"는 동적 프로그래밍 방식
        for amount in range(coin, n+1):         # 현재 동전 coin을 사용해서 만들 수 있는 금액은 coin 이상이므로 coin부터 n까지 반복
            res[amount] += res[amount - coin]   # amount원을 만드는 방법: (amount - coin)원을 만들고 거기에 coin 동전을 추가하는 방식!
                                                # 그러므로 res[amount - coin]만큼의 경우의 수를 res[amount]에 더해준다.
                                                # 기존에 저장되어있는 res[amount - coin]을 불러온다는 점에서 메모이제이션(memoization)과 비슷하다.
    # print(res)
    return res[n]
```

```python
count_coin(1200)
```

```txt
242
```
