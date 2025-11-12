# face_dataset_creator.py
"""
Simple Face Dataset Creator
- Prompts for Name, Roll No, Course
- Creates folder: faces/{Name}_{RollNo}
- Writes meta.json with name, roll_no, course
- Captures images from webcam (default 20 images, 1s apart)
- Press 'q' to stop early
"""

import cv2
import os
import time
import json

def sanitize(s: str) -> str:
    """Make a filesystem-safe single token for folder names."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in s.strip()).strip('_')

def create_dataset(person_name: str, roll_no: str, course: str, images_to_capture: int = 20, interval_sec: float = 1.0):
    safe_name = sanitize(person_name.replace(" ", "_"))
    safe_roll = sanitize(str(roll_no).replace(" ", "_"))
    folder_name = f"{safe_name}_{safe_roll}" if safe_roll else safe_name
    dataset_path = os.path.join('faces', folder_name)
    os.makedirs(dataset_path, exist_ok=True)

    # Save meta.json
    meta = {
        "name": person_name.strip(),
        "roll_no": str(roll_no).strip(),
        "course": course.strip()
    }
    try:
        with open(os.path.join(dataset_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"⚠️ Warning: Failed to write meta.json: {e}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam (tried device 0).")
        print("If you have multiple cameras, try setting CAP device index in the script.")
        return

    print(f"\n🎥 Capturing images for: {person_name} (Roll: {roll_no}, Course: {course})")
    print(f"📁 Saving to: {dataset_path}")
    print(f"ℹ️ Images to capture: {images_to_capture} | Interval: {interval_sec}s")
    print("Press 'q' to stop early.\n")

    img_counter = 0
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame.")
            break

        now = time.time()
        if now - last_capture_time >= interval_sec and img_counter < images_to_capture:
            img_name = os.path.join(dataset_path, f"{folder_name}_{img_counter:03d}.jpg")
            success = cv2.imwrite(img_name, frame)
            if success:
                print(f"✅ Saved [{img_counter+1}/{images_to_capture}]: {os.path.basename(img_name)}")
            else:
                print(f"⚠️ Failed to save: {img_name}")
            img_counter += 1
            last_capture_time = now

            if img_counter >= images_to_capture:
                print("\n🎉 Capture complete!")
                break

        # Overlay info on preview
        info_top = f"{person_name}  |  Roll: {roll_no}  |  Course: {course}"
        info_bottom = f"Captured: {img_counter}/{images_to_capture}  —  Press 'q' to quit"
        cv2.putText(frame, info_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, info_bottom, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # Visual capture indicator
        cv2.circle(frame, (frame.shape[1]-30, 30), 10, (0,0,255) if (img_counter % 2 == 0) else (0,255,0), -1)

        cv2.imshow("Face Dataset Creator - Press 'q' to stop", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print(f"\n⏹️ Stopped by user after capturing {img_counter} images.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✔️ Dataset folder ready: {dataset_path}")
    print("📝 meta.json saved with name, roll_no, course.\n")

if __name__ == "__main__":
    print("\n=== FACE DATASET CREATOR ===")
    name = input("Enter person name (e.g., John Doe): ").strip()
    if not name:
        print("❌ Name cannot be empty. Exiting.")
        exit(1)
    roll = input("Enter roll number (e.g., 12345): ").strip()
    course = input("Enter course (e.g., B.Tech CSE): ").strip()
    # # Optional: allow user to press Enter to accept defaults
    # images_str = input("How many images to capture? [default 20]: ").strip()
    # try:
    #     images_count = int(images_str) if images_str else 20
    # except:
    #     images_count = 20
    # interval_str = input("Interval between captures in seconds? [default 1.0]: ").strip()
    # try:
    #     interval = float(interval_str) if interval_str else 1.0
    # except:
    #     interval = 1.0

    create_dataset(name, roll, course)
