@echo off
REM Add MiKTeX to PATH if it's not already installed system-wide
set "PATH=%PATH%;C:\Users\aweso\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

echo Compiling main.tex 
pdflatex -interaction=nonstopmode main.tex

echo Running biber...
biber main

echo Compilation finished.
pause
