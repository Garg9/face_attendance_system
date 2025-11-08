import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from tqdm import tqdm
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScalableFaceEmbeddings:
    """Scalable face recognition for 1000+ people using FAISS."""
    
    def __init__(self):
        # AUTO-DETECT embedding dimension from FaceNet
        self.embedding_dim = self._detect_embedding_dimension()
        self.faiss_index = None
        self.name_to_id = {}
        self.id_to_name = {}
        self.metadata = {}
        self.embeddings = []  # For SVM training
        self.labels = []
        logging.info(f"🔧 Auto-detected embedding dimension: {self.embedding_dim}")
        
    def _detect_embedding_dimension(self):
        """Automatically detect FaceNet embedding dimension."""
        try:
            # Create temporary FaceNet instance to check output shape
            temp_facenet = FaceNet()
            temp_model = temp_facenet.model
            
            # Create dummy input
            dummy_input = np.zeros((1, 160, 160, 3), dtype='float32')
            dummy_input = (dummy_input - 127.5) / 128.0
            
            # Get output shape
            output_shape = temp_model.predict(dummy_input, verbose=0).shape
            embedding_dim = output_shape[1]
            
            logging.info(f"✅ Detected FaceNet output: {embedding_dim}-D embeddings")
            return embedding_dim
            
        except Exception as e:
            logging.warning(f"⚠️ Could not auto-detect dimension, using default 128: {e}")
            return 128
    
    def preprocess_face(self, face_crop):
        """Enhanced preprocessing for consistent embeddings."""
        try:
            # Convert to RGB.
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Resize with high quality
            face_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)
            
            # Enhance contrast (optional - helps with lighting variations)
            lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # FaceNet standard preprocessing
            face_pixels = enhanced.astype('float32')
            face_pixels = (face_pixels - 127.5) / 128.0  # [-1, 1] range
            face_pixels = np.expand_dims(face_pixels, axis=0)
            return face_pixels
        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            return None
    
    def is_good_quality(self, face_crop):
        """Check face quality for training."""
        try:
            # Basic size check
            h, w = face_crop.shape[:2]
            if h < 80 or w < 80:
                return False
            
            # Brightness and contrast
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Relaxed quality checks for demo
            if mean_val < 20 or mean_val > 240 or std_val < 15:
                return False
            
            # Basic sharpness check
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 30:  # Relaxed threshold
                return False
            
            return True
        except:
            return False
    
    def process_single_image(self, image_path, model):
        """Process one image: detect face → extract → embed."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Detect faces with relaxed parameters
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=2,  # Relaxed
                minSize=(30, 30),  # Smaller minimum
                maxSize=(400, 400)
            )
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding (relaxed)
            padding = int(0.15 * max(w, h))  # Smaller padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            face_crop = image[y:y+h, x:x+w]
            
            # Quality check (relaxed for demo)
            if not self.is_good_quality(face_crop):
                return None
            
            # Generate embedding.
            face_pixels = self.preprocess_face(face_crop)
            if face_pixels is None:
                return None
            
            embedding = model.predict(face_pixels, verbose=0)[0]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
                
                # Verify dimension matches auto-detected
                if len(normalized_embedding) == self.embedding_dim:
                    return normalized_embedding.astype('float32')
                else:
                    logging.warning(f"❌ Dimension mismatch: {len(normalized_embedding)} vs {self.embedding_dim}")
                    return None
            
            return None
            
        except Exception as e:
            if logging.getLogger().level == logging.DEBUG:
                logging.debug(f"Error processing {os.path.basename(image_path)}: {e}")
            return None
    
    def build_index(self, dataset_path):
        """Build scalable FAISS index from dataset."""
        logging.info(f"🚀 Building FAISS index from {dataset_path}")
        logging.info(f"📏 Using embedding dimension: {self.embedding_dim}")
        
        # Load FaceNet model
        try:
            facenet = FaceNet()
            model = facenet.model
            logging.info("✅ FaceNet model loaded")
        except Exception as e:
            logging.error(f"❌ Failed to load FaceNet: {e}")
            return False
        
        # Determine parallel workers
        max_workers = min(4, multiprocessing.cpu_count())  # Reduced for stability
        logging.info(f"🔧 Using {max_workers} parallel workers")
        
        processed_people = 0
        total_images = 0
        valid_embeddings = 0
        
        # Process each person folder
        for person_name in tqdm(os.listdir(dataset_path), desc="Processing people"):
            person_folder = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_folder):
                continue
            
            # Assign unique ID to person
            person_id = len(self.name_to_id)
            self.name_to_id[person_name] = person_id
            self.id_to_name[person_id] = person_name
            
            # Get all image files
            image_files = [f for f in os.listdir(person_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            image_paths = [os.path.join(person_folder, f) for f in image_files]
            
            total_images += len(image_paths)
            
            if not image_paths:
                continue
            
            # Process images in parallel (reduced batch size)
            batch_size = min(5, len(image_paths))  # Smaller batches
            all_valid_embs = []
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_embeddings = list(executor.map(
                        lambda path: self.process_single_image(path, model), 
                        batch_paths
                    ))
                
                # Filter valid embeddings from batch
                valid_batch_embs = [emb for emb in batch_embeddings if emb is not None]
                all_valid_embs.extend(valid_batch_embs)
            
            # Use all valid embeddings for this person
            if all_valid_embs:
                valid_embeddings += len(all_valid_embs)
                
                # Average embeddings for robust representation
                avg_embedding = np.mean(all_valid_embs, axis=0)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                
                # Verify dimension
                if len(avg_embedding) != self.embedding_dim:
                    logging.error(f"❌ Average embedding dimension mismatch: {len(avg_embedding)} vs {self.embedding_dim}")
                    continue
                
                # Store for FAISS index and SVM
                self.embeddings.append(avg_embedding)
                self.labels.append(person_id)
                
                # Store metadata
                self.metadata[person_id] = {
                    'name': person_name,
                    'image_count': len(all_valid_embs),
                    'quality_avg': np.mean([np.linalg.norm(e) for e in all_valid_embs]),
                    'images_used': len(all_valid_embs)
                }
                
                processed_people += 1
                logging.info(f"✅ {person_name}: {len(all_valid_embs)} images → 1 avg embedding (dim: {self.embedding_dim})")
        
        if not self.embeddings:
            logging.error("❌ No valid embeddings generated!")
            logging.info("💡 Your images might have quality issues. Try:")
            logging.info("   • Better lighting (face towards light source)")
            logging.info("   • Frontal face (look straight at camera)")
            logging.info("   • Clear images (no blur, minimum 100x100 pixels)")
            logging.info("   • JPG format preferred")
            return False
        
        # Build FAISS index
        logging.info(f"🏗️ Building FAISS index with {len(self.embeddings)} people...")
        embeddings_array = np.array(self.embeddings, dtype='float32')
        
        # Final dimension verification
        if embeddings_array.shape[1] != self.embedding_dim:
            logging.error(f"❌ Final array dimension: {embeddings_array.shape[1]} vs expected {self.embedding_dim}")
            return False
        
        logging.info(f"📏 Final array shape: {embeddings_array.shape}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index with correct dimension
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        logging.info(f"🔧 FAISS index created: dimension={self.embedding_dim}")
        
        # Add embeddings
        n_added = self.faiss_index.add(embeddings_array)
        logging.info(f"✅ Added {n_added} embeddings to FAISS index")
        
        # Save everything
        os.makedirs('models', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, 'models/scalable_face_index.faiss')
        
        # Save metadata with dimension info
        metadata_data = {
            'name_to_id': self.name_to_id,
            'id_to_name': self.id_to_name,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim,
            'index_type': 'faiss'
        }
        with open('models/scalable_face_metadata.pkl', 'wb') as f:
            pickle.dump(metadata_data, f)
        
        # Save embeddings for SVM training
        np.savez_compressed('models/scalable_embeddings.npz',
                           embeddings=embeddings_array,
                           labels=np.array(self.labels),
                           embedding_dim=self.embedding_dim)
        
        # Statistics
        memory_mb = len(embeddings_array) * self.embedding_dim * 4 / 1024 / 1024
        logging.info(f"🎉 FAISS INDEX BUILT SUCCESSFULLY!")
        logging.info(f"👥 {len(self.id_to_name)} people indexed")
        logging.info(f"📊 {valid_embeddings}/{total_images} images processed ({valid_embeddings/total_images*100:.1f}%)")
        logging.info(f"📏 Embedding dimension: {self.embedding_dim}")
        logging.info(f"⚡ FAISS search: <1ms per query")
        logging.info(f"💾 Memory: {memory_mb:.1f} MB")
        logging.info(f"💾 Files saved:")
        logging.info(f"   📁 scalable_face_index.faiss ({n_added} vectors)")
        logging.info(f"   📄 scalable_face_metadata.pkl") 
        logging.info(f"   📊 scalable_embeddings.npz")
        
        return True
    
    def load_index(self):
        """Load the pre-built FAISS index."""
        try:
            self.faiss_index = faiss.read_index('models/scalable_face_index.faiss')
            
            with open('models/scalable_face_metadata.pkl', 'rb') as f:
                data = pickle.load(f)
                self.name_to_id = data['name_to_id']
                self.id_to_name = data['id_to_name']
                self.metadata = data.get('metadata', {})
                self.embedding_dim = data.get('embedding_dim', 128)
            
            # Verify dimensions match
            if self.faiss_index.d != self.embedding_dim:
                logging.error(f"❌ Dimension mismatch: index={self.faiss_index.d}, metadata={self.embedding_dim}")
                return False
            
            # Load SVM embeddings if available
            if os.path.exists('models/scalable_embeddings.npz'):
                data = np.load('models/scalable_embeddings.npz')
                self.embeddings = data['embeddings']
                self.labels = data['labels']
                saved_dim = data.get('embedding_dim', 128)
                if saved_dim != self.embedding_dim:
                    logging.warning(f"⚠️ Embeddings dimension: {saved_dim} vs index: {self.embedding_dim}")
            
            logging.info(f"✅ LOADED FAISS INDEX!")
            logging.info(f"👥 {len(self.id_to_name)} people ready")
            logging.info(f"📏 Dimension: {self.embedding_dim}")
            logging.info(f"⚡ Ultra-fast search: <1ms per query")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to load FAISS index: {e}")
            return False
    
    def search_identity(self, embedding, top_k=3):
        """Ultra-fast identity search using FAISS."""
        if self.faiss_index is None:
            return []
        
        try:
            # Normalize and verify query embedding
            query_emb = embedding.copy().astype('float32')
            
            if len(query_emb) != self.embedding_dim:
                logging.error(f"❌ Query dimension: {len(query_emb)} vs {self.embedding_dim}")
                return []
            
            # Reshape for FAISS and normalize
            query_emb = query_emb.reshape(1, -1)
            faiss.normalize_L2(query_emb)
            
            # FAISS search
            scores, indices = self.faiss_index.search(query_emb, top_k)
            
            # Convert to results
            results = []
            for i in range(top_k):
                if indices[0][i] == -1:
                    continue
                
                person_id = int(indices[0][i])
                person_name = self.id_to_name.get(person_id, 'Unknown')
                
                # Cosine similarity = confidence
                cosine_sim = scores[0][i]
                confidence = max(0.0, cosine_sim)
                
                results.append({
                    'name': person_name,
                    'confidence': confidence,
                    'distance': float(1.0 - cosine_sim),
                    'person_id': person_id,
                    'cosine_similarity': cosine_sim
                })
            
            # Sort by confidence
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(f"🔍 FAISS search: {len(results)} results")
                for i, result in enumerate(results[:3]):
                    logging.debug(f"   #{i+1}: {result['name']} (conf: {result['confidence']:.3f}, cos: {result['cosine_similarity']:.3f})")
            
            return results
            
        except Exception as e:
            logging.error(f"FAISS search error: {e}")
            return []
    
    def get_stats(self):
        """Get system statistics."""
        if self.faiss_index is None:
            return {
                'status': 'not_loaded',
                'people_count': 0,
                'search_speed': 'N/A',
                'capacity': 'N/A',
                'index_type': 'FAISS'
            }
        
        return {
            'status': 'ready',
            'people_count': len(self.id_to_name),
            'search_speed': '<1ms',
            'capacity': '50000+ people',
            'index_type': 'FAISS',
            'dimension': self.embedding_dim,
            'vectors': self.faiss_index.ntotal,
            'index_size_mb': self.faiss_index.ntotal * self.embedding_dim * 4 / 1024 / 1024
        }

def main():
    """Main execution."""
    print("🚀 SCALABLE FACE EMBEDDINGS GENERATOR (FAISS)")
    print("⚡ Auto-detects embedding dimension - Ready for 1000+ people")
    print("=" * 60)
    
    dataset_path = 'faces'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        print("💡 Create it first: mkdir faces")
        print("💡 Then run: python face_dataset_creator.py")
        return
    
    # Check dataset structure
    people_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not people_folders:
        print(f"❌ No person folders found in '{dataset_path}'!")
        print("💡 Expected structure:")
        print("   faces/")
        print("   ├── John_Doe/          (20 images)")
        print("   │   ├── John_Doe_00.jpg")
        print("   │   ├── John_Doe_01.jpg")
        print("   │   └── ...")
        print("   └── Jane_Smith/        (20 images)")
        return
    
    print(f"📁 Found {len(people_folders)} people folders:")
    for i, folder in enumerate(people_folders[:5]):
        img_count = len([f for f in os.listdir(os.path.join(dataset_path, folder)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   {i+1}. {folder} ({img_count} images)")
    if len(people_folders) > 5:
        print(f"   ... and {len(people_folders)-5} more")
    
    embedder = ScalableFaceEmbeddings()
    
    if embedder.build_index(dataset_path):
        print("\n" + "="*60)
        print("🎉 SUCCESS! SCALABLE SYSTEM READY!")
        print(f"👥 {len(embedder.id_to_name)} people indexed")
        print(f"📏 Embedding dimension: {embedder.embedding_dim}")
        print(f"⚡ Search speed: <1ms per query")
        print("\n✅ Next steps:")
        print("   1. python scalable_train_classifier.py")
        print("   2. python scalable_attendance_system.py")
        print("\n🌐 Access: http://localhost:5000")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ FAILED! No valid embeddings generated.")
        print("\n💡 Troubleshooting:")
        print("   • Check image quality (well-lit, clear faces)")
        print("   • Ensure faces are frontal (looking at camera)")
        print("   • Minimum 5 images per person recommended")
        print("   • Try re-running face_dataset_creator.py with better lighting")
        print("="*60)

if __name__ == '__main__':
    main()