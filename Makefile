DATA ?= data
OUT  ?= out
PY   ?= python

.PHONY: install run serve synth demo test
install:
	$(PY) -m pip install -r requirements.txt
run:
	$(PY) run.py --data $(DATA) --out $(OUT)
serve:
	$(PY) serve.py --out $(OUT)
synth:
	$(PY) tools/make_synthetic.py --out data_synth
demo: synth
	$(PY) run.py --data data_synth --out out_synth
test:
	$(PY) tests/test_pipeline.py
	$(PY) tests/test_assistant.py
