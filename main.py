import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    generate_embeddings,
    translate_role_for_streamlit,
    get_available_models  # Add this import
)
from PIL import Image

st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                          ['ChatBot',
                           'Image Captioning',
                           'Embed text',
                           'Ask me anything'],
                          menu_icon='robot',
                          icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                          default_index=0
                          )

if selected == 'ChatBot':
    model = load_gemini_pro_model()
    
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
    
    st.title("🤖 ChatBot")
    
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)
    
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)
elif selected == "Image Captioning":
    st.title("📷 Snap Narrate")
    
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        if st.button("Generate Caption"):
            try:
                with st.spinner("Analyzing image..."):
                    image = Image.open(uploaded_image)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        resized_img = image.resize((800, 500))
                        st.image(resized_img)
                    
                    with col2:
                        caption = gemini_pro_vision_response("Write a detailed caption for this image", image)
                        st.info(caption)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        st.warning("Please upload an image first")


elif selected == "Embed text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")
    
    if st.button("Get Response"):
        if user_prompt.strip():
            with st.spinner("Generating embeddings..."):
                try:
                    response = generate_embeddings(user_prompt)
                    
                    if isinstance(response, list):
                        st.success("Embeddings generated successfully!")
                        st.write(f"Embedding dimension: {len(response)}")
                        st.write("First 10 values:")
                        st.write(response[:10])
                        
                        # Add download button
                        st.download_button(
                            label="Download embeddings",
                            data=str(response),
                            file_name="embeddings.txt",
                            mime="text/plain"
                        )
                    elif isinstance(response, str) and "Error" in response:
                        st.error(response)
                    else:
                        st.write("Embedding values:", response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text")

elif selected == "Ask me anything":
    st.title("❓ Ask me a question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")
    
    if st.button("Get Response"):
        if user_prompt.strip():
            with st.spinner("Generating response..."):
                response = gemini_pro_response(user_prompt)
                st.markdown(response)
        else:
            st.warning("Please enter a question")