import os
import base64
from pymongo import MongoClient
from pathlib import Path
import json

# MongoDB connection
MONGODB_URI = "mongodb+srv://harilogicgo_db_user:g6Zz4M2xWpr3B2VM@cluster0.bnzjt7f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGODB_URI)
db = client.halloween_db
garments_collection = db.garments

# Path to your Halloween Dress folder
HALLOWEEN_DRESS_PATH = Path("Halloween Dress")

def upload_images_to_mongodb():
    """Upload all images from Halloween Dress folder to MongoDB"""
    
    # Clear existing data
    garments_collection.delete_many({})
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    uploaded_count = 0
    
    for image_file in sorted(HALLOWEEN_DRESS_PATH.iterdir()):
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            try:
                # Read image file
                with open(image_file, "rb") as f:
                    image_data = f.read()
                
                # Convert to base64 for storage
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Create document
                garment_doc = {
                    "filename": image_file.name,
                    "url": f"/garment_templates/{image_file.name}",
                    "image_data": image_base64,
                    "file_size": len(image_data),
                    "content_type": f"image/{image_file.suffix[1:].lower()}"
                }
                
                # Insert into MongoDB
                result = garments_collection.insert_one(garment_doc)
                print(f"[OK] Uploaded: {image_file.name} (ID: {result.inserted_id})")
                uploaded_count += 1
                
            except Exception as e:
                print(f"[ERROR] Error uploading {image_file.name}: {e}")
    
    print(f"\n[SUCCESS] Successfully uploaded {uploaded_count} images to MongoDB!")
    
    # Verify upload
    total_count = garments_collection.count_documents({})
    print(f"[INFO] Total garments in database: {total_count}")
    
    # List all uploaded files
    print("\n[LIST] Uploaded files:")
    for doc in garments_collection.find({}, {"filename": 1, "url": 1}):
        print(f"  - {doc['filename']} -> {doc['url']}")

if __name__ == "__main__":
    upload_images_to_mongodb()
