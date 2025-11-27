@REM Minimal makefile for Sphinx documentation
@REM

@REM You can set these variables from the command line.
set SPHINXOPTS=
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

@REM Put it first so that "make" without argument is like "make help".
help:
	@%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

.PHONY: help Makefile

@REM Catch-all target: route all unknown targets to Sphinx using the new
@REM "make mode" option.
%: Makefile
	@%SPHINXBUILD% -M %* %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

