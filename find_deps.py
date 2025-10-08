#!/usr/bin/env python3
"""
Find all imported packages in Python files, excluding specified directories
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict

# Directories to exclude
EXCLUDED_DIRS = {'eda', 'scripts', '__pycache__', '.git', 'venv', 'env', '.venv', 'node_modules'}

# Specific files to exclude
EXCLUDED_FILES = {'llm_translation_suite_analysis.py', 'run_evaluation.py'}

# Standard library modules (Python 3.11+) - won't be included in requirements
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv', 'ctypes',
    'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis',
    'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler',
    'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib',
    'functools', 'gc', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp',
    'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'imaplib', 'imghdr', 'imp',
    'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword',
    'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap',
    'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt',
    'multiprocessing', 'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse',
    'os', 'ossaudiodev', 'parser', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes',
    'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
    'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue',
    'quopri', 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter',
    'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil',
    'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver',
    'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog',
    'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
    'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize', 'tomllib', 'trace',
    'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
    'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave',
    'weakref', 'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml',
    'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo'
}


def extract_imports_from_file(filepath):
    """Extract all import statements from a Python file"""
    imports = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get the base package name
                    base_package = alias.name.split('.')[0]
                    imports.add(base_package)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the base package name
                    base_package = node.module.split('.')[0]
                    imports.add(base_package)
    
    except SyntaxError:
        print(f"⚠️  Syntax error in {filepath}")
    except Exception as e:
        print(f"⚠️  Error parsing {filepath}: {e}")
    
    return imports


def find_python_files(root_dir, excluded_dirs, excluded_files):
    """Find all Python files excluding specified directories and files"""
    python_files = []
    root_path = Path(root_dir)
    
    for py_file in root_path.rglob('*.py'):
        # Check if file is in an excluded directory
        if any(excluded in py_file.parts for excluded in excluded_dirs):
            continue
        
        # Check if file name is in excluded files
        if py_file.name in excluded_files:
            continue
        
        python_files.append(py_file)
    
    return python_files


def find_dependencies(root_dir='.', excluded_dirs=None, excluded_files=None, show_files=False):
    """
    Find all dependencies in Python files
    
    Args:
        root_dir: Root directory to search
        excluded_dirs: Set of directory names to exclude
        excluded_files: Set of file names to exclude
        show_files: Whether to show which files use each package
    """
    if excluded_dirs is None:
        excluded_dirs = EXCLUDED_DIRS
    if excluded_files is None:
        excluded_files = EXCLUDED_FILES
    
    print(f"Scanning Python files in: {Path(root_dir).absolute()}")
    print(f"Excluding directories: {', '.join(sorted(excluded_dirs))}")
    print(f"Excluding files: {', '.join(sorted(excluded_files))}\n")
    
    # Find all Python files
    python_files = find_python_files(root_dir, excluded_dirs, excluded_files)
    print(f"Found {len(python_files)} Python files\n")
    
    # Extract imports from all files
    all_imports = defaultdict(set)
    
    for py_file in python_files:
        imports = extract_imports_from_file(py_file)
        for imp in imports:
            all_imports[imp].add(py_file)
    
    # Separate standard library from third-party packages
    third_party = {}
    stdlib = {}
    
    for package, files in all_imports.items():
        if package in STDLIB_MODULES:
            stdlib[package] = files
        else:
            third_party[package] = files
    
    # Print results
    print("=" * 70)
    print("THIRD-PARTY PACKAGES (add to requirements.txt)")
    print("=" * 70)
    
    if third_party:
        for package in sorted(third_party.keys()):
            print(f"  • {package}")
            if show_files:
                for file in sorted(third_party[package]):
                    print(f"      - {file}")
        
        print(f"\n📦 Total third-party packages: {len(third_party)}")
        
        # Generate requirements.txt format
        print("\n" + "=" * 70)
        print("REQUIREMENTS.TXT FORMAT")
        print("=" * 70)
        for package in sorted(third_party.keys()):
            print(package)
    else:
        print("  No third-party packages found")
    
    print("\n" + "=" * 70)
    print("STANDARD LIBRARY MODULES (built-in)")
    print("=" * 70)
    
    if stdlib:
        for package in sorted(stdlib.keys()):
            print(f"  • {package}")
            if show_files:
                for file in sorted(stdlib[package]):
                    print(f"      - {file}")
        
        print(f"\n📚 Total standard library modules: {len(stdlib)}")
    else:
        print("  No standard library imports found")
    
    print("\n" + "=" * 70)
    
    return third_party, stdlib


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find Python package dependencies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_deps.py                          # Scan current directory
  python find_deps.py --dir /path/to/project   # Scan specific directory
  python find_deps.py --exclude eda scripts    # Exclude additional directories
  python find_deps.py --exclude-files test.py  # Exclude specific files
  python find_deps.py --show-files             # Show which files use each package
  python find_deps.py --output requirements.txt # Save to file
        """
    )
    
    parser.add_argument(
        '--dir', '-d',
        default='.',
        help='Root directory to scan (default: current directory)'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        default=[],
        help='Additional directories to exclude'
    )
    
    parser.add_argument(
        '--exclude-files', '-x',
        nargs='+',
        default=[],
        help='Additional files to exclude (e.g., test.py config.py)'
    )
    
    parser.add_argument(
        '--show-files', '-f',
        action='store_true',
        help='Show which files use each package'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for requirements.txt format'
    )
    
    args = parser.parse_args()
    
    # Combine default excluded dirs with user-specified ones
    excluded = EXCLUDED_DIRS | set(args.exclude)
    excluded_files = EXCLUDED_FILES | set(args.exclude_files)
    
    # Find dependencies
    third_party, stdlib = find_dependencies(
        root_dir=args.dir,
        excluded_dirs=excluded,
        excluded_files=excluded_files,
        show_files=args.show_files
    )
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            for package in sorted(third_party.keys()):
                f.write(f"{package}\n")
        print(f"\n✅ Saved to: {args.output}")