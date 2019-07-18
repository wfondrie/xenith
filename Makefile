help:
	@echo 'Build the binaries for PyPI upload'
	@echo '    help:     Display this text.'
	@echo '    clean:    Remove old binary builds.'
	@echo '    build:    Build new binaries.'
	@echo '    update:   Update setuptools, wheel, and twine'
	@echo '    docs:     Build the documentation'
	@echo '    install:  Install xenith using `pip install -e .`'
	@echo '    test:     Run tests.'
	@echo '    release:  Run the other commands in the following order:'
	@echo '              clean -> update -> install -> test -> docs -> build'
	@echo '    upload:   Upload to PyPI.'
	@echo ' '
	@echo 'Every new release should run `make release && make upload`.'

clean:
	rm -r dist docs/_build

install:
	pip install -e .

build:
	python3 setup.py sdist bdist_wheel

upload:
	twine upload dist/*

update:
	conda upgrade setuptools wheel twine

test:
	pytest

logo: xenith_logo.svg
	cp xenith_logo.svg docs/_static/

docs: logo
	cd docs && make html

release: clean update install test docs build

.PHONY: help clean build upload update docs release
