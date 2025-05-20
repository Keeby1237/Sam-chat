import os
# Disable Streamlit's file watcher for PyTorch
os.environ["STREAMLIT_WATCHED_MODULES"] = ""

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"


import streamlit as st
import datetime
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MODEL_COLLECTION = [
    "smilyai-labs/Sam-flash-mini-v1",
    "smilyai-labs/Sam-large-v1-speacil",
    "smilyai-labs/Sam-large-v1-speacil-v1-cpu",
    "smilyai-labs/Sam-reason-v1",
    "smilyai-labs/Sam-reason-v2",
    "smilyai-labs/Sam-reason-v3",
    "smilyai-labs/Sam-reason-S1",
    "smilyai-labs/Sam-reason-S2",
    "smilyai-labs/Sam-reason-S3"
]

USER_DB = "user_db.json"
CHAT_DB = "chat_db.json"
MESSAGE_LIMIT = 15
DEVICE = torch.device("cpu")

# Load model on demand
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# User management
def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    return username in users and users[username]["password"] == password

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {"password": password, "last_used": str(datetime.date.today()), "count": 0}
    save_users(users)
    return True

def can_send_message(username):
    users = load_users()
    user = users[username]
    today = str(datetime.date.today())
    if user["last_used"] != today:
        user["last_used"] = today
        user["count"] = 0
    if user["count"] < MESSAGE_LIMIT:
        user["count"] += 1
        save_users(users)
        return True
    return False

# Chat saving/loading
def load_chat(username):
    if os.path.exists(CHAT_DB):
        with open(CHAT_DB, "r") as f:
            data = json.load(f)
            return data.get(username, [])
    return []

def save_chat(username, messages):
    data = {}
    if os.path.exists(CHAT_DB):
        with open(CHAT_DB, "r") as f:
            data = json.load(f)
    data[username] = messages
    with open(CHAT_DB, "w") as f:
        json.dump(data, f)

# Query the model
def query_model(model_name, prompt, max_new_tokens=256, temperature=0.7):
    model, tokenizer = load_model(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result[len(prompt):].strip()

# Streamlit UI
st.set_page_config(page_title="SmilyAI Chat", layout="wide")
st.title("🤖 SmilyAI Chat Interface")

if "username" not in st.session_state:
    st.session_state.username = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "smilyai-labs/Sam-reason-S1"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Login/Register System
with st.sidebar:
    st.header("🔐 Login / Register")
    if st.session_state.username:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            save_chat(st.session_state.username, st.session_state.messages)
            st.session_state.username = None
            st.session_state.messages = []
    else:
        action = st.radio("Choose Action", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button(action):
            if action == "Login":
                if authenticate(username, password):
                    st.session_state.username = username
                    st.session_state.messages = load_chat(username)
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password")
            else:
                if register_user(username, password):
                    st.success("User registered successfully!")
                else:
                    st.error("Username already exists")

# Chat UI
if st.session_state.username:
    with st.sidebar:
        st.markdown("### 🔧 Select Model")
        model_name = st.selectbox("Choose Model", MODEL_COLLECTION, index=MODEL_COLLECTION.index("smilyai-labs/Sam-reason-S1"))
        st.session_state.model_name = model_name

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something...")
    if prompt:
        if can_send_message(st.session_state.username):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = query_model(st.session_state.model_name, prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    save_chat(st.session_state.username, st.session_state.messages)
        else:
            st.warning("Daily message limit reached (15). Try again tomorrow.")
else:
    st.info("Please log in to use the chat.")
