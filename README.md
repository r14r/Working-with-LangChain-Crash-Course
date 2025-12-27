# Helper Libraries Documentation

## Overview

The workspace includes two powerful helper modules to simplify Ollama SDK integration:

- **`helper_ollama`**: Core Ollama SDK functionality wrappers
- **`helper_streamlit`**: Streamlit-specific UI components and integrations

## Installation & Setup

The helpers are located in `/lib` directory. To use them in your pages:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from helper_ollama import OllamaHelper, get_installed_models
from helper_streamlit import select_model, chat_interface, generate_text
```

## helper_ollama Module

### OllamaHelper Class

Main class for Ollama operations:

```python
from helper_ollama import OllamaHelper

helper = OllamaHelper()
```

### Model Management

```python
# List installed models
models = helper.list_models()  # Detailed info
model_names = helper.get_model_names()  # Just names

# Pull/download a model
for progress in helper.pull_model("gemma3", stream=True):
    print(progress['status'])

# Delete a model
result = helper.delete_model("old-model")

# Copy a model
helper.copy_model("gemma3", "my-gemma3")

# Create custom model
helper.create_model(
    name="my-assistant",
    from_model="gemma3",
    system="You are a helpful coding assistant"
)

# Get model details
info = helper.show_model("gemma3")

# List running models
running = helper.list_running_models()

# Check if model exists
if helper.is_model_installed("gemma3"):
    print("Ready to use!")
```

### Chat & Generate

```python
# Chat
response = helper.chat(
    model="gemma3",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False
)

# Async chat
response = await helper.async_chat(
    model="gemma3",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Generate
response = helper.generate(
    model="gemma3",
    prompt="Write a poem",
    stream=True
)

# Chat with tools
response = helper.chat_with_tools(
    model="llama3.1",
    messages=[{"role": "user", "content": "What's 5+3?"}],
    tools=[add_function, subtract_function]
)
```

### Embeddings

```python
# Generate embeddings
result = helper.embed(
    model="nomic-embed-text",
    input_text="Hello world"
)

embeddings = result['embeddings'][0]
```

### Convenience Functions

```python
from helper_ollama import (
    get_installed_models,  # List model names
    get_model_details,     # Detailed model info
    pull_model,            # Pull a model
    delete_model,          # Delete a model
    is_model_installed     # Check if installed
)

# Quick usage
models = get_installed_models()
if not is_model_installed("gemma3"):
    pull_model("gemma3")
```

## helper_streamlit Module

### StreamlitOllamaHelper Class

```python
from helper_streamlit import StreamlitOllamaHelper

helper = StreamlitOllamaHelper()
```

### UI Components

#### Model Selector

```python
# In sidebar
model = helper.render_model_selector(
    key="my_model",
    label="Select Model",
    use_installed=True,  # Use installed models
    location="sidebar"
)

# In main area
model = helper.render_model_selector(
    key="my_model",
    location="main",
    default_models=["gemma3", "llama3.2"]
)
```

#### Model Settings

```python
settings = helper.render_model_settings(
    key_prefix="my_settings",
    show_temperature=True,
    show_max_tokens=True,
    show_top_p=True,
    show_top_k=False,
    location="sidebar"
)

# Use settings in API calls
# settings = {'temperature': 0.7, 'num_predict': 512, 'top_p': 0.9}
```

#### Model Info Display

```python
helper.render_model_info("gemma3", location="sidebar")
```

### Chat Functions

#### Simple Chat

```python
response = helper.run_chat(
    model="gemma3",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    options={'temperature': 0.7}
)
```

#### Chat with History

```python
# Complete chat interface with session management
helper.run_chat_with_history(
    model="gemma3",
    session_key="my_chat",
    initial_messages=[
        {"role": "system", "content": "You are helpful"}
    ],
    options={'temperature': 0.7}
)

# Clear history
helper.clear_chat_history("my_chat")
```

### Generate Functions

```python
response = helper.run_generate(
    model="gemma3",
    prompt="Write a story",
    stream=True,
    options={'temperature': 0.8},
    images=[image_bytes]  # For multimodal
)
```

### Model Management UI

```python
# Render model list
helper.render_model_list()

# Pull model UI
helper.render_model_pull_ui()

# Show running models
helper.render_running_models()
```

### Convenience Functions

```python
from helper_streamlit import (
    select_model,      # Quick model selector
    render_settings,   # Quick settings UI
    chat_interface,    # Complete chat UI
    generate_text      # Quick generation
)

# Minimal chat page
model = select_model(key="model", location="sidebar")
settings = render_settings(location="sidebar")
chat_interface(model=model)
```

## Complete Examples

### Minimal Chat Page

```python
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from helper_streamlit import select_model, chat_interface

st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat with Ollama")

# Model selection in sidebar
model = select_model(key="chat_model", location="sidebar")

# Chat interface
chat_interface(model=model, session_key="chat_history")
```

### Generate Page with Settings

```python
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from helper_streamlit import select_model, render_settings, generate_text

st.title("Text Generation")

# Sidebar
model = select_model(location="sidebar")
settings = render_settings(
    location="sidebar",
    show_temperature=True,
    show_max_tokens=True
)

# Main area
prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    response = generate_text(
        model=model,
        prompt=prompt,
        stream=True,
        options=settings
    )
```

### Model Management Page

```python
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from helper_streamlit import StreamlitOllamaHelper

st.title("Model Management")

helper = StreamlitOllamaHelper()

tab1, tab2, tab3 = st.tabs(["Installed", "Running", "Download"])

with tab1:
    helper.render_model_list()

with tab2:
    helper.render_running_models()

with tab3:
    helper.render_model_pull_ui()
```

## Benefits

- **Less Boilerplate**: Reusable UI components
- **Consistent UX**: Standardized widgets across pages
- **Easy Model Management**: Built-in model operations
- **Session Management**: Automatic chat history handling
- **Streaming Support**: Built-in streaming visualization
- **Error Handling**: Graceful error management
- **Type Hints**: Full type annotations for IDE support

## Tips

1. **Always add lib to path** at the top of your page
2. **Use convenience functions** for quick implementations
3. **Customize with parameters** for specific needs
4. **Check examples** in `99_🔧_Helper_Example.py`
5. **Extend the helpers** by subclassing when needed

See `pages/99_🔧_Helper_Example.py` for live demonstrations!
