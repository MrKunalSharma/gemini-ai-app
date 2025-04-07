import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from PIL import Image

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def translate_role_for_streamlit(role):
    """Translate Gemini role to Streamlit role"""
    if role.lower() == "model":
        return "assistant"
    return role.lower()

def load_gemini_pro_model():
    """Load Gemini Pro model"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model
    except Exception as e:
        raise Exception(f"Error loading Gemini Pro model: {str(e)}")
    
    
def gemini_pro_vision_response(prompt, image):
    """Get response from Gemini Pro Vision model"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating image caption: {str(e)}"

def get_available_models():
    """Get list of available models"""
    try:
        available_models = genai.list_models()
        vision_models = []
        for model in available_models:
            if 'vision' in model.name.lower():
                # Only include non-deprecated models
                if 'gemini-1.0' not in model.name:
                    model_name = model.name.replace('models/', '')
                    vision_models.append(model_name)
        return vision_models
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        return []

def generate_embeddings(text):
    """Generate embeddings using Gemini AI"""
    try:
        # Initialize the embedding model
        model = 'models/embedding-001'
        
        # Generate embeddings
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        
        # Extract embeddings from response
        if isinstance(response, dict) and 'embedding' in response:
            return response['embedding']
        elif hasattr(response, 'embedding'):
            return response.embedding
        else:
            return list(response)
        
    except Exception as e:
        return f"Error generating embeddings: {str(e)}"

def gemini_pro_response(user_prompt):
    """Get response from Gemini Pro model"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"