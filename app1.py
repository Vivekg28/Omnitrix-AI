import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import ollama

# --- PAGE CONFIG ---
st.set_page_config(page_title="Omnitrix AI", page_icon="⌚", layout="wide")

# Custom Omnitrix Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #00ff00;
    }
    h1, h2, h3 {
        color: #39ff14 !important;
    }
    div.stButton > button {
        background-color: #00ff00;
        color: black;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⌚ Ben 10 Omnitrix RAG")
st.write("Multimodal AI: Text-to-Alien & Alien-to-Text")

# --- 1. SETUP DATABASE (Cached) ---
@st.cache_resource
def setup_database():
    IMAGE_FOLDER = "aliens"
    CSV_FILE = "aliens.csv"
    COLLECTION_NAME = "omnitrix_web_db"

    embedding_func = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()
    client = chromadb.Client()
    
    try: client.delete_collection(COLLECTION_NAME)
    except: pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        data_loader=data_loader
    )

    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
    except:
        st.error(f"Could not read {CSV_FILE}")
        return None

    ids = []
    image_paths = []
    metadatas = []

    for index, row in df.iterrows():
        name = str(row['name']).strip().lower()
        # Check png, jpg, jpeg
        img_path_png = os.path.join(IMAGE_FOLDER, f"{name}.png")
        img_path_jpg = os.path.join(IMAGE_FOLDER, f"{name}.jpg")
        img_path_jpeg = os.path.join(IMAGE_FOLDER, f"{name}.jpeg")
        
        final_path = None
        if os.path.exists(img_path_png): final_path = img_path_png
        elif os.path.exists(img_path_jpg): final_path = img_path_jpg
        elif os.path.exists(img_path_jpeg): final_path = img_path_jpeg
        
        if final_path:
            ids.append(name)
            image_paths.append(final_path)
            metadatas.append({
                "name": str(row['name']),
                "species": str(row.get('species', 'Unknown')), 
                "powers": str(row.get('powers', 'Unknown'))
            })
    
    if ids:
        collection.add(ids=ids, uris=image_paths, metadatas=metadatas)
        return collection
    return None

with st.spinner("Initializing Omnitrix Core..."):
    collection = setup_database()

if not collection:
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["🔍 Search Alien (Text-to-Image)", "🧬 Scan Alien (Image-to-Text)"])

# --- TAB 1: TEXT TO IMAGE (UPDATED) ---
with tab1:
    st.header("Select Alien from Description")
    
    # 1. Initialize Session State to hold results
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Describe the Alien (e.g. 'Red fire guy')", "")
    with col2:
        st.write("") # Spacer
        st.write("")
        search_clicked = st.button("Search DNA")

    # 2. Perform Search when Button Clicked
    if search_clicked and query:
        results = collection.query(
            query_texts=[query],
            n_results=3, # Get Top 3
            include=['uris', 'metadatas']
        )
        # Save to session state so it remembers
        if results['uris'] and results['uris'][0]:
            st.session_state.search_results = results
        else:
            st.warning("No matches found.")
            st.session_state.search_results = None

    # 3. Display Selection List (If results exist)
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        # Create a list of names for the radio button
        # We store them as "Name (Species)"
        options = []
        for i in range(len(results['uris'][0])):
            meta = results['metadatas'][0][i]
            options.append(f"{meta['name']} ({meta['species']})")
            
        st.write("---")
        st.subheader("Found these matches:")
        
        # The Selection Widget
        selected_option = st.radio("Choose one to view:", options)
        
        # 4. Show the ONE selected image
        if selected_option:
            # Find the index of the selected option
            idx = options.index(selected_option)
            
            # Get data for that specific index
            img_path = results['uris'][0][idx]
            data = results['metadatas'][0][idx]
            
            st.write("---")
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                st.image(img_path, width=300)
            
            with col_info:
                st.markdown(f"## **{data['name']}**")
                st.markdown(f"**Species:** `{data['species']}`")
                st.markdown(f"**Powers:** {data['powers']}")

# --- TAB 2: IMAGE TO TEXT ---
with tab2:
    st.header("Upload Alien Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
        with col2:
            st.info("Scanning DNA...")
            query_image = Image.open("temp_query.jpg").convert("RGB")
            query_array = np.array(query_image)
            
            results = collection.query(
                query_images=[query_array],
                n_results=1,
                include=['metadatas']
            )
            
            if results['metadatas'] and results['metadatas'][0]:
                match = results['metadatas'][0][0]
                st.success(f"✅ IDENTITY: {match['name'].upper()}")
                st.write(f"**Species:** {match['species']}")
                st.write(f"**Powers:** {match['powers']}")
                
                st.write("---")
                st.write("🤖 **Vision Analysis:**")
                with st.spinner("Analyzing..."):
                    try:
                        response = ollama.chat(
                            model='moondream',
                            messages=[{
                                'role': 'user',
                                'content': f"Describe this image of {match['name']}.",
                                'images': ["temp_query.jpg"]
                            }]
                        )
                        st.write(response['message']['content'])
                    except:
                        st.warning("Ollama not responding.")
            else:
                st.error("Unknown DNA.")