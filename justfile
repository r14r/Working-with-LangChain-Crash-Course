default:
	@just -l

install:
	uv sync
	
serve:
	uv run streamlit run Intro.py

run: serve

