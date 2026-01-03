
import streamlit as st
from pathlib import Path

def build_navigation():
    """
    Build navigation structure by scanning pages subfolders.
    Groups pages by topic derived from subfolder names.
    Icons are shown only in navigation group items, not in filenames.

    Returns:
        List of st.Page objects organized by topic sections
    """
    pages_dir = Path(__file__).parent / "pages"

    # Topic configuration with display names and emojis for group navigation
    # Maps folder names (with numeric prefix) to display info
    topics = {
        "00_home": {"name": "🏠 Home", "icon": "🏠", "order": 0},
        "01_learn": {"name": "📚 Learn", "icon": "📚", "order": 1},
        "02_projects": {"name": "🚀 Projects", "icon": "🚀", "order": 2},
        "11_beginner": {"name": "🌱 Beginner", "icon": "🌱", "order": 11},
        "12_advanced": {"name": "🌿 Intermediate", "icon": "🌿", "order": 12},
        "13_expert": {"name": "🌳 Advanced", "icon": "🌳", "order": 13},
    }

    # Map specific pages to their individual icons
    page_icons = {
        "Thinking": "🧠",
        "Thinking_Generate": "🧠",
        "Thinking_Levels": "🧠",
        "Fill_in_Middle": "💻"
    }

    # Build navigation structure
    navigation_pages = {}

    # Scan each topic subfolder
    for topic_folder, topic_info in sorted(topics.items(), key=lambda x: x[1]["order"]):
        topic_path = pages_dir / topic_folder

        if not topic_path.exists():
            continue

        print(f"Scanning topic folder: {topic_folder}")

        # Get all Python files in this topic folder
        py_files = sorted(topic_path.glob("*.py"))

        if py_files:
            pages = []
            for py_file in py_files:
                # Extract title from filename (remove number prefix)
                title = py_file.stem.split("_", 1)[1] if "_" in py_file.stem else py_file.stem
                title = title.replace("_", " ")

                # Get icon for specific pages, otherwise use topic icon
                icon = page_icons.get(py_file.stem.split("_", 1)[1] if "_" in py_file.stem else "",
                                     topic_info["icon"])
                icon = None

                # Create st.Page for each script
                page = st.Page(
                    str(py_file),
                    title=title,
                    icon=icon
                )
                pages.append(page)

                print(f"  Found page: {title} (icon: {icon})")

            # Add this topic's pages to navigation
            if pages:
                navigation_pages[topic_info["name"]] = pages

    return navigation_pages


# -----

# --- Navigation ---

pages = build_navigation()

pg = st.navigation(pages)
pg.run()