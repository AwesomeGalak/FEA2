@echo off
REM Add MiKTeX to PATH if it's not already installed system-wide
set "PATH=%PATH%;C:\Users\aweso\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

echo Compiling main.tex (1/3)...
pdflatex -interaction=nonstopmode main.tex

echo Running biber...
biber main

echo Compiling main.tex (2/3)...
pdflatex -interaction=nonstopmode main.tex

echo Compiling main.tex (3/3)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo Compilation finished.
pause
