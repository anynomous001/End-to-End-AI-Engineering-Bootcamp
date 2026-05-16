import streamlit as st
import requests
from chatbot_ui.core.config import config


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

st.title("Shopping Assistant Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with products today?"}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

# --- Sidebar logic ---
st.sidebar.markdown("<h3 style='color: #ff4b4b; border-bottom: 2px solid #ff4b4b; padding-bottom: 5px; margin-bottom: 20px;'>Suggestions</h3>", unsafe_allow_html=True)

# Create a placeholder in the sidebar for dynamic rendering
sidebar_placeholder = st.sidebar.empty()

def render_suggestions():
    with sidebar_placeholder.container():
        for item in st.session_state.used_context:
            st.markdown(f"<small>{item.get('description', '')}</small>", unsafe_allow_html=True)
            if item.get("image_url"):
                st.image(item["image_url"], use_container_width=True)
            if item.get("price"):
                st.caption(f"Price: {item['price']} USD")
            st.divider()

# Initial render of sidebar suggestions (on reload)
render_suggestions()

# --- Main chat logic ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you with products today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        success, response_data = api_call("post", f"{config.API_URL}/api/v1/rag/", json={"query": prompt})
        
        if success:
            answer = response_data.get("answer", "No answer received.")
            st.session_state.used_context = response_data.get("used_context", [])
            render_suggestions()  # Dynamically update the sidebar
        else:
            answer = response_data.get("message", "An error occurred while connecting to the API.")
            
        st.write(answer)
        
    st.session_state.messages.append({"role": "assistant", "content": answer})