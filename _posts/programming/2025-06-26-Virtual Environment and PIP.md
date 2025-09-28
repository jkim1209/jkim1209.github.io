---
layout: post
title: "파이썬 가상환경 설정과 PIP"
date: 2025-06-26
categories: [Programming, Python]
tags: [python, virtual environment, venv, pip, package management, development environment, python setup]
---

## 가상환경(Virtual Environment)

프로젝트에서 사용할 가상환경을 만들어 봅시다. 파이썬 개발을 위한 가상환경을 만들면 다른 프로젝트에 영향을 주지 않고 작업할 수 있습니다.  
예를 들어 A라는 프로젝트에는 오픈AI라이브러리의 0.28버전을 사용하고, B라는 프로젝트에서는 1.0버전을 사용하는 상황을 가정해봅시다.  
만약 가상환경을 만들지 않으면 오픈AI 라이브러리를 1.0 버전으로 설치할 때 A 프로젝트의 기존 설정이 영향을 받을 수 있습니다.  
라이브러리 버전에 따라 동작이 달라지거나 사용하는 방법이 변경될 수 있으므로 A프로젝트에서 잘 동작하던 코드가 더 이상 실행되지 않는 문제가 생길 수 있습니다.  
생성형 AI관련 라이브러리와 패키지는 몇 주 단위로 업데이트되고 있습니다.  
오픈 AI라이브러리 역시 2023년 11월 업데이트되면서 사용법이 크게 변경되었습니다.  
0.28버전에서 잘 작동하던 코드도 1.0버전을 설치한 이후에는 오류가 발생했죠.
가상 환경을 구축하면 이런 문제를 방지할 수 있습니다.  

---

### 1. 가상환경 폴더 생성

VS Code에서 프로젝트 폴더로 사용할 폴더를 엽니다. 그리고 `Ctrl` + '`'을 눌러 터미널 창을 열고 다음과 같이 가상환경 이름을 입력하세요.

`$ python -m venv '가상환경명'`

```powershell
python -m venv venv
```

#### 1.1. **버전을 지정해서 설치하고 싶은 경우**  

(단, 해당 파이썬 버젼이 컴퓨터에 설치되어 있어야 합니다.)  

`$ py -버전 -m venv '가상환경명'`

```powershell
py -3.8.8 -m venv venv
```

#### 1.2. 다른 드라이브에 파이썬이 설치되어 있는 등 Windows Py Launcher가 해당 Python을 감지하지 못하는 경우

직접 Python 실행경로를 써주면 됩니다.

`$ '설치하고싶은 파이썬경로가 설치된 경로'\python.exe -m venv '가상환경명'`

```powershell
C:\Python\Python310\python.exe -m venv venv310
```

실행하면 왼쪽 프로젝트 폴더에 '가상환경명'의 폴더가 생깁니다.  
이 폴더 안에 가상 환경을 위한 파일이 준비됩니다.

생성된 후에 해당 가상환경에서의 파이썬 버전이 궁금하다면 다음과 같이 명령어를 입력하면 됩니다.  

`$ '가상환경명'\Scripts\python --version`

```powershell
venv310\Scripts\python --version
```

#### 1.3. 현재폴더가 아닌 곳에 설치하고 싶은 경우  

`$ python -m venv '설치하고싶은 폴더경로\가상환경명'`

```powershell
python -m venv C:\Python\Python310\venv310
```

(참고)  파이썬 버전이 설치된 폴더 내에서 venv를 생성하면 해당 버전으로 가상환경을 구성하므로 현재 파이썬 버전과 상관없이 파이썬 3.10버전의 가상환경이 생성됩니다.

#### 1.4 `1.2`와 `1.3`의 내용을 종합  

가령 "VS Code에서 D드라이브의 프로젝트 폴더를 작업하면서도, 실제 가상환경(venv)은 C드라이브에 만들어서 그걸 사용하고 싶다"면 다음과 같이 명령어를 실행하면 됩니다.  

`$  '설치하고싶은 파이썬경로가 설치된 경로'\python.exe -m venv '설치하고싶은 폴더경로\가상환경명'`

```powershell
C:\Python\Python310\python.exe -m venv C:\Venvs\venv310
```

### 2. 터미널 활성화

VS Code의 터미널 창에 다음과 같이 입력해 가상환경을 활성화합니다.  

`$ .\'가상환경명'\Scripts\activate`

```powershell
.\venv\Scripts\activate
```

### 2.1 가상환경이 현재 폴더에 있지 않은 경우  

가상환경의 경로까지 적어주면 됩니다.  

`$ '가상환경이 설치되어 있는 폴더경로\가상환경명'\Scripts\activate`

```powershell
C:\Venvs\venv310\Scripts\activate
```

경로 앞에 (`가상환경명`)이 표시되면 성공입니다. 이제 앞으로 모든 작업은 가급적 이 가상환경을 만들어 활성화한 상태에서 해주세요.

### 2.2. jupyter notebook 의 경우  

우측 상단의 Select Kernel을 클릭했을 때 자동으로 가상환경을 찾아준다면 좋지만, 목록에 없다면  
① 우선 Select Kernel을 클릭 후 사용할 가상환경의 Python 버전과 같은 로컬의 Python을 클릭해줍니다.  
② VS Code에서 `Ctrl`+`Shift`+`P` -> `Python: Select Interpreter` -> `Enter interpreter path` -> `Find` 를 선택한 뒤  
③ 생성된  `가상환경이 설치되어 있는 폴더경로\가상환경명`\Scripts\python.exe 를 수동으로 찾아 선택하면 됩니다.  
④ 이후 우측 상단의 Select Kernel을 클릭 후 `Select Another Kernel` -> `Python Environments' 를 클릭해주면 해당 가상환경이 목록에 있습니다.  
(① 을 거치지 않아도 되는 경우도 있습니다. 현재 내 폴더와 가상환경이 다른 공간에 있다면 가상환경을 최근에 불러왔느냐 아니냐가 Select Kernel에서 보이는지 여부를 결정하는 것 같습니다.)  

---

**보안오류: UnauthorizedAccess 가 발생한다면?**

(1) 윈도우 창에 'Windows PowerShell'을 검색하고 [관리자로 실행]  
(2) 다음을 입력  

```powershell
PS C:\WINDOWS\system32> get-ExecutionPolicy
```

현재 restricted로 되어있다면 스크립트 실행이 허용되지 않는 상태를 의미합니다.
실행정책을 변경하기 위해 Set-ExecutionPolicy RemoteSigned를 입력하고 이어서 y를 입력합니다.  

```powershell
PS C:\WINDOWS\system32> get-ExecutionPolicy
Restricted
PS C:\WINDOWS\system32> Set-ExecutionPolicy RemoteSigned
실행 규칙 변경
...
...
...
변경하시겠습니까?
[Y] 예(Y) [A] 모두 예(A) [N] 아니요(N) [L] 모두 아니요(L) [S] 일시 중단(S) [?] 도움말
(기본값은 "N"): y
```

이제 VS Code 터미널 창에 .\`가상환경명`\Scripts\activate 를 입력하면 오류 없이 가상환경이 설정됩니다.

---

> **NOTE**:   필자의 경우 여러 버전의 Python과 그 버전들을 이용한 가상환경들을 효율적으로 사용하기 위해 Python 폴더들은 C드라이브 하위에 `Python`폴더를 생성하여 그 안에 설치하였고,  
>
> * (e.g. Python312의 경로: C:\Python\Python312)  
>
> 가상환경 폴더들은 C 드라이브 하위에 `Venvs`폴더를 생성하여 그 안에 설치하고자 합니다.
>
> * (e.g. Python310 버전의 가상환경은 C:\Venvs\venv310 이라는 이름으로 생성  
>
> **NOTE**:   이하 파이썬 코드들을 jupyter notebook에서 실행할경우 pip앞에는 %를, python 혹은 py앞에는 !를 붙여야 합니다.

### 3. 가상환경 빠져나오기

가상환경은 deactivate로 빠져나올 수 있습니다.

```python
deactivate
```

---

**Appendix**: `virtualenv`  

라이브러리 패키지인 virtualenv를 이용해서 가상환경을 만들수도 있습니다.
기본적으로 venv와 virtualenv 둘 다 가상 환경을 만드는 라이브러리지만, 약간의 차이가 있습니다.

* venv: Python 3.3 버전 이후부터 기본 라이브러리로 포함되어 별도의 설치 과정이 필요없음.  
* virtualenv: Python 2 버전부터 쓰던 라이브러리로, Python 3에서도 사용 가능하고 별도의 설치 과정 필요.  

더 정확하게 venv 모듈은 virtualenv의 경량화된 모듈로, 속도와 확장성 측면에서 virtualenv이 더 우수합니다.  
대신 venv는 기본 내장 라이브러리이기 때문에 pip install의 설치 과정이 필요없어서 간단합니다.  

① virtualenv를 먼저 설치

```python
pip install virtualenv
```

② virtualenv 가상환경 생성  

`$ virtualenv '가상환경명'`

```python
virtualenv myenv
```

* **여러 개 파이썬 설치 버전 중 버전 지정해서 설하고 싶은 경우**  

`$ virtualenv 가상환경이름 --python=파이썬 버전`

```python
virtualenv venv --python=3.10.11
```

③ 가상환경 활성화 및 종료는 venv와 동일  

`$ .\'가상환경명'\Scripts\activate`  
`$ deactivate`  

---

## 다른 버전의 파이썬 가상환경 추가로 설치

### 1. 파이썬 버전과 설치 경로 확인

만약 아나콘다를 설치했다면 아나콘다에 기본으로 포함된 파이썬 버전이 깔려있는 것을 볼 수 있습니다.
이는 `where python`, `which python` 명령어를 통해 파이썬이 설치된 모든 경로를 볼 수 있습니다.

### 2. 특정 버전의 파이썬을 따로 설치하여 특정 버전의 파이썬이 설치된 폴더에서 가상환경을 실행하기

(1) 다른 버전의 파이썬 다운로드  
[https://www.python.org/downloads/](https://www.python.org/downloads/) 에서 특정 버전의 파이썬 다운로드합니다.  

(2) 다운로드된 파일을 관리자 권한으로 실행하고, 설치할 때 Add Python `버전` to PATH는 반드시 체크합니다.  
> Tip)    Customize installation을 선택하여 C 드라이브 안에 Python 폴더를 하나 만들고 그 안에서 파이썬 버전들을 관리하는 것이 나중에 용이하기 때문에 아래와 같이 설치 경로를 변경하고 설치하시길 권장합니다.  
>
> * Customize install location: `C:\Python\Python'버전'`

(3) 파이썬 버전 확인해보기  
이제 cmd창이나 PowerShell에서 python 명령어를 통해 설치한 파이썬 버전을 확인해봅니다.  
방금 설치한 파이썬 버전이 아닌 기존의 파이썬 버전이 실행됩니다.  
이유는 Add Python `버전` to PATH 를 체크하여 설치를 진행했는데, 설치한 파이썬 버전의 환경 변수 PATH가 이전 파이썬 버전의 PATH보다 실행 우선 순위가 밀렸기 때문입니다.  
그러나 새롭게 설치한 파이썬 버전이 설치된 폴더 (위처럼 진행했다면 `C:\Python\Python'버전'`)에서 python명령어를 실행하면 정상적으로 새롭게 설치한 버전이 실행됩니다.  
현재 위치를 기준으로 실행 파일을 검색하기 때문에 당연한 결과입니다. 그러나 매번 실행 파일이 설치된 폴더의 경로를 찾아서 실행하기 번거롭기 때문에 PATH를 설정하는 것입니다.  

(4) 환경변수 PATH확인  
[`고급 시스템 설정 보기` - `환경변수`]에서 `변수 Path`를 클릭 후 편집에 들어가면, 우리가 설치한 파이썬 버전의 경로가 기존 파이썬 버전의 경로보다 아래에 위치해 있습니다.  
python 파일이 두 폴더 모두 존재한다면, 우선 순위가 높은(위쪽에 위치한) 폴더의 파일을 실행합니다.  
만약 새롭게 설치한 파이썬 버전을 내 컴퓨터의 기본값으로 사용하고 싶으면, 해당 PATH를 위로 이동시킵니다.  

(5) 파이썬 설치 폴더 내에서 가상환경 구성  
환경변수 path와 상관없이 파이썬 버전이 설치된 폴더 내에서 venv를 생성하면 해당 버전으로 가상환경을 구성합니다.

---

## pip

파이썬으로 작성된 각종 라이브러리를 설치하고 관리해주는 도구입니다.

### 1. pip 업그레이드, 버젼확인

pip는 자주 업데이트되므로 가상환경에서 최신 버전의 pip로 업그레이드해주는 것이 좋습니다.  

```python
pip install --upgrade pip
pip --version
```

> 단, 가상환경이 아닌 로컬에서 진행할 때는 한 줄 추가하여 다음과 같이 pip를 업그레이드 하면 좋습니다.  
> `$ python -m pip install --user --upgrade pip` : 현재 사용자 계정에 pip  
> 파이썬 패키지 관리자를 최신 버전으로 업그레이드하는 명령어입니다.

```python
pip install --upgrade pip
python -m pip install --user --upgrade pip
```

### 2. 라이브러리 설치/업그레이드/삭제

기본적으로 버전 정보가 입력되지 않으면 최신버전이 설치됩니다.  
`$ pip install 패키지명==버전넘버(e.g.,2.3.0)`  

특정 버전 이상을 설치  
`$ pip install 패키지명>=버전넘버(e.g.,2.3.0)`  

특정 라이브러리 업그레이드  
`$ pip install --upgrade 패키지명`  

라이브러리 삭제  
`$ pip uninstall 패키지명`

라이브러리 정보 확인  
`$ pip show 패키지명`

### 3. pip 설치 리스트 확인  

현재 (가상)환경 내에 설치된 라이브러리 목록 나열  

```python
pip list
```

### 4. 현재 패키지를 다른 (가상)환경에 설치하기

freeze명령어는 pip install에 맞는 형태로 패키지 목록을 *.txt 파일에 저장합니다. (파일명은 변경 가능)  
프로젝트 협업 시 동일한 작업 환경과 버전을 보장할 수 있습니다.  

```python
pip freeze > requirements.txt
```

이제 패키지를 설치하고 싶은 다른 가상 환경에 가서 다음의 명령어를 실행합니다.  

```python
pip install -r requirements.txt
```

---

## Reference

> Do it! LLM을 활용한 AI 에이전트 개발 입문
