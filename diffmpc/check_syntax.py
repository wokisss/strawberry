import py_compile
import os
import traceback
import sys

has_error = False
for dp, dn, fs in os.walk('.'):
    for f in fs:
        if f.endswith('.py'):
            path = os.path.join(dp, f)
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as e:
                print(f"Syntax Error in {path}:\n{e}")
                has_error = True

if not has_error:
    print("NO SYNTAX ERRORS FOUND")
