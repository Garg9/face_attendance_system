import cv2
import os
import time

def create_dataset(person_name):
    """Create face dataset for one person."""
    # Create the directory for the person if it doesn't exist
    dataset_path = f'faces/{person_name}'
    os.makedirs(dataset_path, exist_ok=True)
    
    # Initialize webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print(f"🎥 Webcam opened. Capturing 20 images for {person_name}...")
    print("💡 Tips: Good lighting, face camera, varied expressions")
    print("Press 'q' to stop early.")

    img_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame.")
            break
            
        current_time = time.time()
        
        # Capture a photo every 2 seconds
        if current_time - start_time >= 2:
            img_name = f"{dataset_path}/{person_name}_{img_counter:02d}.jpg"
            success = cv2.imwrite(img_name, frame)
            if success:
                print(f"✅ Saved: {person_name}_{img_counter:02d}.jpg")
            else:
                print(f"⚠️ Failed to save: {img_name}")
            img_counter += 1
            start_time = current_time

            if img_counter >= 20:
                print("🎉 20 images captured! Dataset complete.")
                break
        
        # Display the frame with visual cues
        if img_counter < 20:
            remaining = 20 - img_counter
            cv2.putText(frame, f"Capturing: {remaining} left", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {person_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Red circle indicator
        cv2.circle(frame, (frame.shape[1] - 40, 30), 15, (0, 0, 255), -1)
        cv2.circle(frame, (frame.shape[1] - 40, 30), 15, (255, 255, 255), 2)

        cv2.imshow("🎥 SCALABLE Dataset Creator - Press 'q' to stop", frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print(f"⏹️ Stopped by user after {img_counter} images.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📁 Dataset saved to: faces/{person_name}/")

if __name__ == '__main__':
    print("🚀 SCALABLE FACE DATASET CREATOR")
    print("📝 For 1000+ people: Run this script for each person")
    print("-" * 50)
    
    name = input("Enter person name (e.g., John_Doe): ").strip()
    if not name:
        print("❌ Name cannot be empty!")
    else:
        create_dataset(name.replace(" ", "_"))
        print(f"\n✅ Complete! Next: python scalable_face_embeddings.py")