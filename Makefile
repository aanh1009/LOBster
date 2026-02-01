.PHONY: install test benchmark demo lint clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/test_engine.py tests/test_microstructure.py -v

benchmark:
	python benchmarks/benchmark_lob.py

demo:
	python run_demo.py --duration 600 --no-dashboard

demo-full:
	python run_demo.py --duration 1800

lint:
	python -m py_compile engine/order.py engine/order_book.py \
		hawkes/process.py microstructure/metrics.py \
		feed/simulator.py backtest/engine.py run_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
