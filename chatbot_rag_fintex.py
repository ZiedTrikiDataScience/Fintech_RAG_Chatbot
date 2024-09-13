import streamlit as st
from rag_financial_chatbot import search_similar_question, generate_enhanced_answer
import os
from PIL import Image


# Function to handle the chatbot logic
def chatbot(prompt):
    api_key = os.environ.get("MISTRAL_API_KEY")
    #api_key = os.environ["MISTRAL_API_KEY"] = 'joukTqVPkc1Z7XI34QIE2vmEGyncaNsy'
    faq = search_similar_question(prompt)
    enhanced_answer = generate_enhanced_answer(prompt, faq['answer'], api_key=api_key)
    return enhanced_answer

# Load any assets (like logos) for UI enhancements
def load_assets():
    logo = Image.open('fintech.jpg')  # Add a logo image for your chatbot
    return logo

# Streamlit interface setup
def app():
    # Load assets
    logo = load_assets()

    # Sidebar with settings
    st.sidebar.header("Settings")
    st.sidebar.subheader("Chatbot Customization")
    temperature = st.sidebar.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150)

    # Display the logo in the main page
    st.image(logo, width=200)
    
    # Page Title and Description
    st.title("FinTechX FAQ Chatbot 🤖")
    st.markdown("""
        Welcome to the **FinTechX FAQ Chatbot**. 
        Ask any question about **FinTechX**, and the chatbot will provide an answer using **Retrieval-Augmented Generation (RAG)**, 
        powered by **Mistral AI**.
        """)

    # User input field
    user_query = st.text_input("Ask a question about FinTechX", placeholder="e.g., How do I reset my password?")

    # Provide some sample questions to guide the user
    st.markdown("**Sample Questions**")
    st.write("• What is the withdrawal limit?")
    st.write("• How can I reset my password?")
    st.write("• How do I contact customer support?")

    # Display chatbot response
    if user_query:
        with st.spinner("Generating response..."):
            response = chatbot(user_query)
            st.success("Here's the answer:")
            st.write(f"**Answer:** {response}")

        # Optionally store the conversation history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({"query": user_query, "response": response})

        # Display conversation history
        st.markdown("### Conversation History")
        for idx, interaction in enumerate(st.session_state.history, 1):
            st.write(f"**Q{idx}:** {interaction['query']}")
            st.write(f"**A{idx}:** {interaction['response']}")

# Run the app
if __name__ == "__main__":
    app()
