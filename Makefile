# Linux / macOS / CI.  ON WINDOWS USE .\run.ps1 INSTEAD -- see below.
#
# Windows has no `make`, PowerShell has no equivalent, and under WSL this
# file's $(HOME) resolves to the Linux home rather than C:\Users\<you> (and a
# WSL shell cannot execute the Windows venv anyway). Rather than make you
# install a toolchain to run three commands, run.ps1 is the supported entry
# point there and mirrors every target below:
#
#     .\run.ps1 help        .\run.ps1 install     .\run.ps1 test
#     .\run.ps1 fit         .\run.ps1 run         .\run.ps1 learn
#
# The venv lives OUTSIDE the repo on purpose. This project sits ~90 characters
# deep, Windows caps paths at 260, and torch ships licence files with
# ~160-character relative paths -- an in-repo .venv fails to install torch
# with WinError 206. Enabling long paths system-wide is the alternative:
#   Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
#
# Override the location if you want:  make VENV=.venv install
VENV ?= $(HOME)/.venvs/linesight
URL  ?= http://192.168.68.100:8080/video

# Scripts/ on Windows, bin/ everywhere else.
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
else
PY := $(VENV)/bin/python
endif

.PHONY: help venv install test test-fast lint fmt fit calibrate report probe clean tape check align learn inspect whitepaper

help:
	@echo "Windows? use .\run.ps1 help instead."
	@echo "make venv       create the virtualenv"
	@echo "make install    install linesight and its dependencies (editable)"
	@echo "make test       run the full suite"
	@echo "make test-fast  skip anything marked slow or data"
	@echo "make lint       ruff check"
	@echo "make fit        build the memory bank for AITEX fabric 02"
	@echo "make calibrate  derive its threshold from a false-alarm budget"
	@echo "make report     render the latest stored roll to PDF"
	@echo "make probe P=p04_tiling   run one probe"

venv:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: venv
	$(PY) -m pip install -e ".[api,report,dev]" --extra-index-url https://download.pytorch.org/whl/cpu

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow and not data"

lint:
	$(PY) -m ruff check src tests probes

fmt:
	$(PY) -m ruff format src tests probes
	$(PY) -m ruff check --fix src tests probes

fit:
	$(PY) -m linesight fit --sku aitex_02 --normal data/aitex/normal_02/

calibrate:
	$(PY) -m linesight calibrate --sku aitex_02 --clean data/aitex/clean_02/ --budget 1.0

report:
	$(PY) -m linesight report --roll latest --out results/roll_report.pdf

# make probe P=p02_memory_bank
probe:
	$(PY) probes/$(P).py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# --- bench rig -------------------------------------------------------------- #
# The rig commands. On Windows use .\run.ps1 tape|check|align|learn|inspect.
tape:
	$(PY) tools/make_tape.py --length-m 2 --ids

check:
	$(PY) probes/p13_phone_stream.py --url $(URL) --sku bench

align:
	$(PY) tools/align_view.py --url $(URL) --sku bench

learn:
	$(PY) tools/bench_run.py learn --url $(URL) --sku bench

inspect:
	$(PY) tools/bench_run.py inspect --url $(URL) --sku bench --pdf results/bench_report.pdf

whitepaper:
	$(PY) tools/make_whitepaper.py
