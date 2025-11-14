---
layout: post
title: "Python Virtual Environment Setup and PIP"
date: 2025-06-26
categories: [Programming, Python]
tags: [python, virtual environment, venv, pip, package management, development environment, python setup]
---

## Virtual Environment

Let's create a virtual environment for your project. Creating a virtual environment for Python development allows you to work without affecting other projects.
For example, suppose project A uses version 0.28 of the OpenAI library, while project B uses version 1.0.
If you don't create a virtual environment, installing the OpenAI library at version 1.0 may affect the existing setup of project A.
Code that worked well in project A may no longer run because the behavior changes or usage methods are modified depending on the library version.
Libraries and packages related to generative AI are being updated every few weeks.
The OpenAI library was also significantly changed in the November 2023 update.
Code that worked well in version 0.28 caused errors after installing version 1.0.
Building a virtual environment can prevent these problems.

---

### 1. Creating a Virtual Environment Folder

Open the folder you'll use as your project folder in VS Code. Then press `Ctrl` + '`' to open the terminal window and enter the virtual environment name as follows.

`$ python -m venv 'environment_name'`

```powershell
python -m venv venv
```

#### 1.1. **If you want to install with a specific version**

(However, that Python version must be installed on your computer.)

`$ py -version -m venv 'environment_name'`

```powershell
py -3.8.8 -m venv venv
```

#### 1.2. If Windows Py Launcher doesn't detect Python (e.g., Python installed on a different drive)

You can directly specify the Python execution path.

`$ 'path_to_python_installation'\python.exe -m venv 'environment_name'`

```powershell
C:\Python\Python310\python.exe -m venv venv310
```

When executed, a folder named 'environment_name' will be created in the left project folder.
Files for the virtual environment are prepared inside this folder.

If you're curious about the Python version in the created virtual environment, you can enter the following command:

`$ 'environment_name'\Scripts\python --version`

```powershell
venv310\Scripts\python --version
```

#### 1.3. If you want to install in a location other than the current folder

`$ python -m venv 'target_folder_path\environment_name'`

```powershell
python -m venv C:\Python\Python310\venv310
```

(Note) If you create a venv inside the folder where a Python version is installed, it will configure the virtual environment with that version, so a Python 3.10 virtual environment will be created regardless of the current Python version.

#### 1.4 Combining contents of `1.2` and `1.3`

For example, if you want to "work on a project folder on D drive in VS Code but create and use the virtual environment (venv) on C drive," you can execute the command as follows:

`$ 'path_to_python_installation'\python.exe -m venv 'target_folder_path\environment_name'`

```powershell
C:\Python\Python310\python.exe -m venv C:\Venvs\venv310
```

### 2. Activating the Terminal

Activate the virtual environment by entering the following in the VS Code terminal window:

`$ .\'environment_name'\Scripts\activate`

```powershell
.\venv\Scripts\activate
```

### 2.1 If the virtual environment is not in the current folder

You need to include the path to the virtual environment.

`$ 'virtual_environment_installation_folder_path\environment_name'\Scripts\activate`

```powershell
C:\Venvs\venv310\Scripts\activate
```

If (`environment_name`) appears at the beginning of the path, you've succeeded. From now on, please do all work in this activated virtual environment as much as possible.

### 2.2. For jupyter notebook

If Select Kernel at the top right automatically finds the virtual environment, that's great, but if it's not in the list:
① First click Select Kernel and click on the local Python that matches the Python version of the virtual environment you want to use.
② In VS Code, press `Ctrl`+`Shift`+`P` -> `Python: Select Interpreter` -> `Enter interpreter path` -> select `Find`, then
③ Manually find and select `virtual_environment_installation_folder_path\environment_name`\Scripts\python.exe.
④ After that, click Select Kernel at the top right, then click `Select Another Kernel` -> `Python Environments', and the virtual environment will be in the list.
(In some cases, you don't need to go through ①. If your current folder and virtual environment are in different locations, whether you recently loaded the virtual environment seems to determine whether it's visible in Select Kernel.)

---

**If you get a security error: UnauthorizedAccess?**

(1) Search for 'Windows PowerShell' in the Windows search and [Run as administrator]
(2) Enter the following:

```powershell
PS C:\WINDOWS\system32> get-ExecutionPolicy
```

If it's currently set to restricted, it means script execution is not allowed.
To change the execution policy, enter Set-ExecutionPolicy RemoteSigned and then enter y.

```powershell
PS C:\WINDOWS\system32> get-ExecutionPolicy
Restricted
PS C:\WINDOWS\system32> Set-ExecutionPolicy RemoteSigned
Execution Policy Change
...
...
...
Do you want to change?
[Y] Yes  [A] Yes to All  [N] No  [L] No to All  [S] Suspend  [?] Help
(default is "N"): y
```

Now when you enter .\`environment_name`\Scripts\activate in the VS Code terminal window, the virtual environment will be set up without errors.

---

> **NOTE**: In my case, to efficiently use multiple Python versions and virtual environments using those versions, I created a `Python` folder under the C drive and installed the Python folders inside it,
>
> * (e.g., Python312 path: C:\Python\Python312)
>
> For virtual environment folders, I want to create a `Venvs` folder under the C drive and install them inside it.
>
> * (e.g., Create a virtual environment for Python310 version with the name C:\Venvs\venv310)
>
> **NOTE**: When executing the following Python code in jupyter notebook, you need to add % before pip and ! before python or py.

### 3. Exiting the Virtual Environment

You can exit the virtual environment with deactivate.

```python
deactivate
```

---

**Appendix**: `virtualenv`

You can also create a virtual environment using the virtualenv library package.
Basically, both venv and virtualenv are libraries that create virtual environments, but there are slight differences.

* venv: Included as a standard library from Python 3.3 onwards, so no separate installation process is required.
* virtualenv: A library used since Python 2, also usable in Python 3, but requires a separate installation process.

More precisely, the venv module is a lightweight version of virtualenv, and virtualenv is superior in terms of speed and extensibility.
However, venv is a built-in standard library, so it's simpler because it doesn't require the pip install installation process.

① Install virtualenv first

```python
pip install virtualenv
```

② Create virtualenv virtual environment

`$ virtualenv 'environment_name'`

```python
virtualenv myenv
```

* **If you want to specify a version among multiple installed Python versions**

`$ virtualenv environment_name --python=python_version`

```python
virtualenv venv --python=3.10.11
```

③ Activating and exiting the virtual environment is the same as venv

`$ .\'environment_name'\Scripts\activate`
`$ deactivate`

---

## Installing Additional Python Virtual Environments with Different Versions

### 1. Check Python Version and Installation Path

If you've installed Anaconda, you'll see the Python version included by default with Anaconda installed.
You can see all paths where Python is installed with the `where python` or `which python` commands.

### 2. Install a specific version of Python separately and run the virtual environment from the folder where that specific Python version is installed

(1) Download a different version of Python
Download a specific version of Python from [https://www.python.org/downloads/](https://www.python.org/downloads/).

(2) Run the downloaded file with administrator privileges, and be sure to check Add Python `version` to PATH during installation.
> Tip) It's recommended to select Customize installation, create a Python folder in the C drive, and change the installation path as follows because it's easier to manage Python versions later:
>
> * Customize install location: `C:\Python\Python'version'`

(3) Check the Python version
Now check the installed Python version through the python command in cmd or PowerShell.
The existing Python version runs instead of the just-installed Python version.
The reason is that even though you installed with Add Python `version` to PATH checked, the environment variable PATH of the installed Python version has lower execution priority than the previous Python version's PATH.
However, if you run the python command in the folder where the newly installed Python version is installed (if you followed the above, `C:\Python\Python'version'`), the newly installed version will run normally.
This is a natural result because it searches for the executable file based on the current location. However, since it's cumbersome to find and execute the path of the folder where the executable file is installed every time, we set the PATH.

(4) Check environment variable PATH
If you click on `variable Path` in [`View advanced system settings` - `Environment Variables`] and enter edit, the path of the Python version we installed is located below the existing Python version's path.
If the python file exists in both folders, it executes the file in the folder with higher priority (located higher).
If you want to use the newly installed Python version as the default for your computer, move that PATH upward.

(5) Configure virtual environment within the Python installation folder
Regardless of the environment variable path, if you create a venv inside the folder where the Python version is installed, it configures the virtual environment with that version.

---

## pip

A tool that installs and manages various libraries written in Python.

### 1. Upgrading pip, Checking version

Since pip is frequently updated, it's good to upgrade to the latest version of pip in the virtual environment.

```python
pip install --upgrade pip
pip --version
```

> However, when doing this in the local environment rather than a virtual environment, it's good to add one line and upgrade pip as follows:
> `$ python -m pip install --user --upgrade pip`: This is a command to upgrade the pip Python package manager to the latest version for the current user account.

```python
pip install --upgrade pip
python -m pip install --user --upgrade pip
```

### 2. Installing/Upgrading/Removing Libraries

By default, if version information is not entered, the latest version is installed.
`$ pip install package_name==version_number(e.g.,2.3.0)`

Install a specific version or higher
`$ pip install package_name>=version_number(e.g.,2.3.0)`

Upgrade a specific library
`$ pip install --upgrade package_name`

Remove a library
`$ pip uninstall package_name`

Check library information
`$ pip show package_name`

### 3. Check pip installation list

List the currently installed libraries in the (virtual) environment

```python
pip list
```

### 4. Installing Current Packages in Another (Virtual) Environment

The freeze command saves the package list in a *.txt file in a format suitable for pip install. (The file name can be changed)
When collaborating on a project, you can ensure the same working environment and version.

```python
pip freeze > requirements.txt
```

Now go to another virtual environment where you want to install the packages and execute the following command:

```python
pip install -r requirements.txt
```

---

## Other Python Environment Management Tools and Comparison

| Category        | Conda                                                                         | venv                                               | uv                                                                                     |
| ----------- | ----------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Basic Concept   | Package + environment management system. Can install not only Python but also other libraries/binaries | Python standard library feature that isolates only Python environments | Ultra-fast package/environment management tool. Implemented in Rust, providing very fast installation and isolation support compared to pip/venv |
| Installation Scope   | Can include Python, libraries, C libraries, etc.                                 | Can only install Python packages (libraries)              | Python package-centric. Fast installation of PyPI packages, ensuring environment reproducibility based on lockfile          |
| Ease of Use | Easy and fast installation on various OS, easy environment management                                  | Available with just Python installed, lightweight         | Very fast installation speed, pip/venv compatible. Few additional tool dependencies                                |
| Package Installation | `conda install package` or `pip install package`                              | `pip install package`                               | `uv add package` or `uv pip install package`                                           |
| Environment Creation   | `conda create -n env_name python=3.11`                                        | `python -m venv env_name`                          | `uv venv env_name`                                                                     |
| Activation      | `conda activate env_name`                                                     | `source env_name/bin/activate` (Mac/Linux)         | `source env_name/bin/activate` (same structure as venv)                                      |
| Features        | Suitable for data science, ML, AI projects                                           | Suitable for simple Python projects, lightweight environments           | Ultra-fast installation, highly reproducible environment management, suitable for large-scale projects/CI                          |

---

## Reference

> Do it! Introduction to AI Agent Development Using LLM
