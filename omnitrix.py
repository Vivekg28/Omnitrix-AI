import os
import pandas as pd
import numpy as np
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import ollama
from tkinter import filedialog
import tkinter as tk

# --- CONFIGURATION ---
IMAGE_FOLDER = "aliens"       # Folder with Heatblast, XLR8 images
CSV_FILE = "aliens.csv"       # File with Name, Species, Powers
COLLECTION_NAME = "omnitrix_db"

print("⌚ Powering up the Omnitrix...")

# 1. SETUP DATABASE
# OpenCLIP handles the translation between Text <-> Images
embedding_func = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()
client = chromadb.Client()

# Reset DB for fresh start
try: client.delete_collection(COLLECTION_NAME)
except: pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    data_loader=data_loader
)

# 2. LOAD & MERGE DATA
print("🧬 Scanning Alien DNA (CSV + Images)...")
try:
    df = pd.read_csv(CSV_FILE)
    # Clean column names (Name -> name, Species -> species)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
except:
    print(f"❌ Error: Could not read {CSV_FILE}")
    exit()

# Critical check for the Name column
if 'name' not in df.columns:
    print("❌ Critical Error: CSV must have a 'Name' column.")
    print(f"   Found columns: {list(df.columns)}")
    exit()

ids = []
image_paths = []
metadatas = []

# Loop through CSV and find matching images
for index, row in df.iterrows():
    name = str(row['name']).strip().lower()
    
    # Check for png, jpg, jpeg
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
        
        # Save Alien Stats
        metadatas.append({
            "name": str(row['name']),
            "species": str(row.get('species', 'Unknown')), 
            "powers": str(row.get('powers', 'Unknown'))
        })
    else:
        print(f"   ⚠️ Warning: Image not found for {row['name']}")

if not ids:
    print("❌ No matches found! Check that image filenames match CSV names.")
    exit()

print(f"   ✅ Locked in {len(ids)} Aliens. Indexing DNA...")

# Add to ChromaDB
collection.add(
    ids=ids,
    uris=image_paths, 
    metadatas=metadatas
)
print("💚 Omnitrix Ready!")

# --- FUNCTIONS ---

def text_to_image():
    """Type text -> Get Top 3 Aliens"""
    query = input("\n📝 Describe the Alien (e.g. 'Fire guy', 'Fast blue alien'): ")
    print("🔍 Searching Database...")
    
    # RETRIEVE TOP 3
    results = collection.query(
        query_texts=[query],
        n_results=3, 
        include=['uris', 'metadatas']
    )
    
    if not results['uris'] or not results['uris'][0]:
        print("No alien found matching that description.")
        return

    print(f"\n🎯 Top Matches for '{query}':")
    
    candidates = []
    # Print options
    count = len(results['uris'][0])
    for i in range(count):
        name = results['metadatas'][0][i]['name']
        img_path = results['uris'][0][i]
        species = results['metadatas'][0][i]['species']
        
        candidates.append(img_path)
        print(f"   {i+1}. {name} ({species})")

    # User Selection
    selection = input("\nWHICH ONE? (Type Number): ")
    
    if selection.isdigit() and 1 <= int(selection) <= count:
        idx = int(selection) - 1
        data = results['metadatas'][0][idx]
        print(f"\n🎉 It's Hero Time! Selected: {data['name']}")
        print(f"   Species: {data['species']}")
        print(f"   Powers: {data['powers']}")
        try:
            Image.open(candidates[idx]).show()
        except:
            print("Could not open image.")
    else:
        print("Invalid selection.")

def image_to_text():
    """Upload Image -> Identify Alien"""
    print("\n🖼️ Select an Alien image to scan...")
    root = tk.Tk()
    root.withdraw()
    query_path = filedialog.askopenfilename()
    
    if not query_path: return
    print(f"👀 Scanning {os.path.basename(query_path)}...")
    
    # 1. VISUAL SEARCH (Pixel Matching)
    try:
        query_image = Image.open(query_path).convert("RGB")
        query_array = np.array(query_image)
    except:
        print("❌ Error loading image.")
        return

    print("🔍 Comparing DNA with Omnitrix database...")
    results = collection.query(
        query_images=[query_array], 
        n_results=1,
        include=['metadatas', 'distances']
    )
    
    if not results['metadatas'] or not results['metadatas'][0]:
        print("❌ DNA Unrecognized.")
        return

    match_data = results['metadatas'][0][0]
    
    print(f"\n✅ IDENTITY CONFIRMED: {match_data['name'].upper()}")
    print(f"   Species: {match_data['species']}")
    print(f"   Powers: {match_data['powers']}")
    
    # 2. GENERATION (Moondream Vision)
    print("\n   Asking AI for tactical analysis...")
    try:
        vision_response = ollama.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': f"Describe this image of Ben 10's alien {match_data['name']}. What is he doing?",
                'images': [query_path]
            }]
        )
        print(f"   📝 Analysis: {vision_response['message']['content']}")
    except:
        print("   (Vision AI not responding, but identity is confirmed)")

# --- MAIN LOOP ---
while True:
    print("\n" + "="*30)
    choice = input("1. Search Alien (Text-to-Image)\n2. Scan Alien (Image-to-Text)\nq. Quit\n> ")
    
    if choice == '1': text_to_image()
    elif choice == '2': image_to_text()
    elif choice == 'q': break