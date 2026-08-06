"""BullaRullaEP: a from-scratch, episodic-pivot-only daily scanner.

Lives inside the bullerulle repo but is otherwise self-contained -- headless
(no Streamlit UI of its own; the parent app's "Browse all charts" tab is
still there if you want to eyeball a candidate), driven by two scheduled
GitHub Actions passes per trading day (see .github/workflows/ep_scan_*.yml)
that push ntfy.sh notifications rather than requiring anyone to have a page
open. See bullarullaep/cli.py for the headless entry points.
"""
