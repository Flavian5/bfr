init:
	pip install -r requirements.txt

test:
	nosetests tests --nocapture

lint:
	pylint ./bfr/*.py ||: