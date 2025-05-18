import streamlit as st
import openai
import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer

# Streamlit UI setup
st.set_page_config(page_title="Recipe Assistant", page_icon="🥘")
st.title("🥘 Recipe Assistant")

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

with st.sidebar:
    st.subheader("Settings")
    openai.api_key = st.text_input("OpenAI API key", type="password")

# Load FAISS index and recipe data
index = faiss.read_index("vector_store/recipes.index")
df = pd.read_csv("vector_store/recipes_meta.csv")

# Retrieval function
def retrieve(query, top_k=5):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [df.iloc[i] for i in indices[0]]

# Prompt generation with context
def generate_prompt(user_input, retrieved_recipes):
    context = "\n\n".join([
        f"Title: {r['Title']}\nIngredients: {r['Ingredients']}\nInstructions: {r['Instructions']}"
        for r in retrieved_recipes
    ])
    prompt = f"""You are a helpful recipe assistant.
Based on the following user query: "{user_input}", recommend a recipe.

Here are some related recipes:
{context}

Answer:"""
    return prompt



# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful recipe assistant."}]

# Display chat history
for msg in st.session_state.messages[1:]:  # Skip the system message
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for chat
user_input = st.chat_input("Ask for a recipe, e.g., 'I want something with eggplant and broccoli'...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Perform retrieval
    retrieved = retrieve(user_input)
    full_prompt = generate_prompt(user_input, retrieved)

    # Add the generated context-augmented prompt to the conversation
    st.session_state.messages.append({"role": "user", "content": full_prompt})

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",  # You can change to "gpt-3.5-turbo" if needed
                    messages=st.session_state.messages
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"Error: {e}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
