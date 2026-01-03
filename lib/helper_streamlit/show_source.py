"""
Show Source Code Helper for Streamlit

This module provides a function to display the source code of the current script.
"""

import streamlit as st
import inspect
import sys
from pathlib import Path


def show_source(file_path: str = None):
    """
    Display the source code of the current script in a Streamlit code block.
    
    Args:
        file_path: Optional path to the file to display. If None, uses the calling script.
    """
    try:
        if file_path is None:
            # Get the calling script's file path
            frame = inspect.currentframe()
            caller_frame = frame.f_back.f_back  # Go back two frames to get the actual caller
            file_path = caller_frame.f_globals.get('__file__')
        
        if file_path:
            # Read the source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Display the source code with syntax highlighting
            st.code(source_code, language='python', line_numbers=True)
        else:
            st.error("Could not determine the source file path.")
            
    except Exception as e:
        st.error(f"Error reading source code: {e}")
