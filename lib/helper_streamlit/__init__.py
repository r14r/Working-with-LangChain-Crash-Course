"""
Streamlit Helper Functions for Ollama

This module provides Streamlit-specific helper functions for integrating
Ollama models into Streamlit applications.
"""

import sys
import os

from typing import List, Dict, Any, Optional

import streamlit as st

from ollama import chat as ollama_chat, generate as ollama_generate

from lib.helper_ollama import OllamaHelper


class StreamlitOllamaHelper:
    """Helper class for Ollama integration in Streamlit"""

    def __init__(self):
        self.ollama = OllamaHelper()

    # ==================== UI Components ====================

    def render_model_selector(
        self,
        key: str = "model_selector",
        label: str = "Select Model",
        default_models: Optional[List[str]] = None,
        use_installed: bool = True,
        location: str = "sidebar",
    ) -> str:
        """
        Render a model selection dropdown in Streamlit

        Args:
            key: Unique key for the selectbox
            label: Label for the selectbox
            default_models: List of default model names to show if none installed
            use_installed: Whether to use installed models or default list
            location: "sidebar" or "main" - where to render the selector

        Returns:
            Selected model name
        """
        models = None

        if use_installed:
            models = self.ollama.get_model_names()

        if not models:
            models = default_models or ["gemma3", "llama3.2", "llama3.1", "qwen2.5"]

        # Determine default index - check if previous selection exists and is still valid
        default_index = 0
        if key in st.session_state and st.session_state[key] in models:
            default_index = models.index(st.session_state[key])

        if location == "sidebar":
            with st.sidebar:
                selected = st.selectbox(label, models, index=default_index, key=key)
        else:
            selected = st.selectbox(label, models, index=default_index, key=key)
        
        return selected

    def render_model_settings(
        self,
        key_prefix: str = "settings",
        show_temperature: bool = True,
        show_max_tokens: bool = True,
        show_top_p: bool = False,
        show_top_k: bool = False,
        location: str = "sidebar",
    ) -> Dict[str, Any]:
        """
        Render common model settings controls

        Args:
            key_prefix: Prefix for widget keys
            show_temperature: Show temperature slider
            show_max_tokens: Show max tokens input
            show_top_p: Show top_p slider
            show_top_k: Show top_k slider
            location: "sidebar" or "main"

        Returns:
            Dictionary of settings
        """
        settings = {}

        container = st.sidebar if location == "sidebar" else st

        with container:
            if show_temperature:
                settings["temperature"] = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    key=f"{key_prefix}_temperature",
                    help="Higher values make output more random",
                )

            if show_max_tokens:
                settings["num_predict"] = st.number_input(
                    "Max Tokens",
                    min_value=10,
                    max_value=4096,
                    value=512,
                    step=50,
                    key=f"{key_prefix}_max_tokens",
                    help="Maximum number of tokens to generate",
                )

            if show_top_p:
                settings["top_p"] = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    key=f"{key_prefix}_top_p",
                    help="Nucleus sampling parameter",
                )

            if show_top_k:
                settings["top_k"] = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=40,
                    key=f"{key_prefix}_top_k",
                    help="Number of top tokens to consider",
                )

        return settings

    def render_model_info(self, model_name: str, location: str = "sidebar"):
        """
        Render model information in an expander

        Args:
            model_name: Model name to show info for
            location: "sidebar" or "main"
        """
        container = st.sidebar if location == "sidebar" else st

        with container:
            with st.expander(f"ℹ️ {model_name} Info"):
                info = self.ollama.show_model(model_name)

                if "error" in info:
                    st.error(info["error"])
                else:
                    if info.get("details"):
                        details = info["details"]
                        st.write(f"**Family:** {details.get('family', 'N/A')}")
                        st.write(f"**Format:** {details.get('format', 'N/A')}")
                        st.write(
                            f"**Parameters:** {details.get('parameter_size', 'N/A')}"
                        )
                        st.write(
                            f"**Quantization:** {details.get('quantization_level', 'N/A')}"
                        )

    # ==================== Chat Functions ====================

    def run_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        container: Optional[Any] = None,
    ) -> str:
        """
        Run a chat interaction and display results

        Args:
            model: Model name
            messages: List of message dictionaries
            stream: Whether to stream the response
            options: Optional model options (temperature, etc.)
            container: Optional Streamlit container to render in

        Returns:
            Complete response text
        """
        if container is None:
            container = st

        if stream:
            response_placeholder = container.empty()
            full_response = ""

            kwargs = {"model": model, "messages": messages, "stream": True}
            if options:
                kwargs["options"] = options

            for chunk in ollama_chat(**kwargs):
                full_response += chunk["message"]["content"]
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            return full_response
        else:
            kwargs = {"model": model, "messages": messages, "stream": False}
            if options:
                kwargs["options"] = options

            response = ollama_chat(**kwargs)
            content = response["message"]["content"]
            container.write(content)
            return content

    def run_chat_with_history(
        self,
        model: str,
        session_key: str = "chat_history",
        initial_messages: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Run a chat interface with session-based history

        Args:
            model: Model name
            session_key: Session state key for chat history
            initial_messages: Initial message history
            options: Optional model options
        """
        # Initialize chat history in session state
        if session_key not in st.session_state:
            st.session_state[session_key] = initial_messages or []

        # Display chat history
        for msg in st.session_state[session_key]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if user_input := st.chat_input("Your message..."):
            # Add user message
            st.session_state[session_key].append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.write(user_input)

            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    kwargs = {"model": model, "messages": st.session_state[session_key]}
                    if options:
                        kwargs["options"] = options

                    response = ollama_chat(**kwargs)
                    assistant_message = response["message"]["content"]
                    st.write(assistant_message)

                    # Add to history
                    st.session_state[session_key].append(
                        {"role": "assistant", "content": assistant_message}
                    )

    def clear_chat_history(self, session_key: str = "chat_history"):
        """Clear chat history from session state"""
        if session_key in st.session_state:
            st.session_state[session_key] = []

    # ==================== Generate Functions ====================

    def run_generate(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List] = None,
        container: Optional[Any] = None,
    ) -> str:
        """
        Run text generation and display results

        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream
            options: Optional model options
            images: Optional images for multimodal models
            container: Optional container to render in

        Returns:
            Generated text
        """
        if container is None:
            container = st

        if stream:
            response_placeholder = container.empty()
            full_response = ""

            kwargs = {"model": model, "prompt": prompt, "stream": True}
            if options:
                kwargs["options"] = options
            if images:
                kwargs["images"] = images

            for chunk in ollama_generate(**kwargs):
                full_response += chunk["response"]
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            return full_response
        else:
            kwargs = {"model": model, "prompt": prompt, "stream": False}
            if options:
                kwargs["options"] = options
            if images:
                kwargs["images"] = images

            response = ollama_generate(**kwargs)
            content = response["response"]
            container.write(content)
            return content

    # ==================== Model Management UI ====================

    def render_model_list(self):
        """Render a table of installed models"""
        models = self.ollama.list_models()

        if not models:
            st.warning("No models installed")
            return

        st.write(f"**Found {len(models)} model(s)**")

        for model in models:
            with st.expander(f"📦 {model['name']}", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Size:** {model['size_mb']:.2f} MB")
                    if model["details"]:
                        st.write(f"**Format:** {model['details']['format']}")
                        st.write(f"**Family:** {model['details']['family']}")

                with col2:
                    if model["details"]:
                        st.write(
                            f"**Parameters:** {model['details']['parameter_size']}"
                        )
                        st.write(
                            f"**Quantization:** {model['details']['quantization_level']}"
                        )

    def render_model_pull_ui(self):
        """Render UI for pulling models"""
        st.subheader("Download Model")

        col1, col2 = st.columns([3, 1])

        with col1:
            model_name = st.text_input(
                "Model name",
                placeholder="e.g., gemma3, llama3.2, codellama",
                key="pull_model_name",
            )

        with col2:
            st.write("")
            st.write("")
            pull_btn = st.button("Pull", key="pull_model_btn")

        if pull_btn and model_name:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                for progress in self.ollama.pull_model(model_name, stream=True):
                    status = progress.get("status", "")
                    if status:
                        status_text.text(f"Status: {status}")

                    total = progress.get("total", 0)
                    completed = progress.get("completed", 0)

                    if total > 0:
                        pct = int((completed / total) * 100)
                        progress_bar.progress(min(pct, 100))

                progress_bar.progress(100)
                st.success(f"✅ Successfully pulled {model_name}!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    def render_running_models(self):
        """Render list of currently running models"""
        models = self.ollama.list_running_models()

        if not models:
            st.info("No models currently running")
            return

        st.write(f"**{len(models)} model(s) running**")

        for model in models:
            with st.expander(f"🚀 {model['name']}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Size:** {model['size_mb']:.2f} MB")
                    st.write(f"**VRAM:** {model['size_vram_mb']:.2f} MB")

                with col2:
                    st.write(f"**Context:** {model['context_length']}")
                    if model["expires_at"]:
                        st.write(f"**Expires:** {model['expires_at']}")


# Convenience instance
_helper = StreamlitOllamaHelper()

# Convenience functions
# Select Model", ["gemma3", "llama3.1", "llama3.2", "qwen2.5"], index=0)


def select_model(
    key: str = "model",
    location: str = "sidebar",
    label="Select Model",
    use_installed=True,
    **kwargs,
) -> str:
    return _helper.render_model_selector(key=key, location=location, label=label, use_installed=use_installed, **kwargs)


def render_settings(
    key_prefix: str = "settings", location: str = "sidebar", **kwargs
) -> Dict[str, Any]:
    return _helper.render_model_settings(
        key_prefix=key_prefix, location=location, **kwargs
    )


def chat(model: str, session_key: str = "chat_history", **kwargs):
    return _helper.run_chat_with_history(model=model, session_key=session_key, **kwargs)


def generate_text(model: str, prompt: str, stream: bool = True, **kwargs) -> str:
    return _helper.run_generate(model=model, prompt=prompt, stream=stream, **kwargs)
