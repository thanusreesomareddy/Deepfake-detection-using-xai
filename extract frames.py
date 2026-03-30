import os
import cv2
import glob
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - CHANGE THIS TO YOUR PATH
# ============================================================================
DATASET_PATH = r"D:\machine learning project\FaceForensics++_C23"  # Your videos path
OUTPUT_PATH = r"D:\faceforensics_frames"  # Where to save frames
FRAMES_PER_VIDEO = 10  # Extract 10 frames per video
FACE_SIZE = 224  # Size for face crop

print("=" * 60)
print("FACEFORENSICS++ FRAME EXTRACTOR")
print("=" * 60)
print(f"Videos path: {DATASET_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Frames per video: {FRAMES_PER_VIDEO}")
print("=" * 60)

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'real'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'fake'), exist_ok=True)

# ============================================================================
# FACE DETECTION FUNCTION
# ============================================================================

def detect_face(frame):
    """Detect and crop face from frame using OpenCV"""
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add margin
        margin = int(0.2 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        # Crop face
        face = frame[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
        
        return face
    return None

# ============================================================================
# EXTRACT FRAMES FROM VIDEOS
# ============================================================================

def extract_frames_from_videos():
    """Extract frames from all videos in dataset"""
    
    total_frames = 0
    video_count = 0
    
    # Process REAL videos (original folder)
    real_videos = glob.glob(os.path.join(DATASET_PATH, 'original', '**', '*.mp4'), recursive=True)
    print(f"\n📹 Found {len(real_videos)} REAL videos")
    
    for video_path in tqdm(real_videos, desc="Processing REAL videos"):
        video_count += 1
        video_name = os.path.basename(video_path).replace('.mp4', '')
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames at regular intervals
        frame_indices = np.linspace(0, total_frames_video-1, FRAMES_PER_VIDEO, dtype=int)
        
        for idx, frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Detect and crop face
                face = detect_face(frame)
                
                if face is not None:
                    # Save face
                    output_path = os.path.join(OUTPUT_PATH, 'real', f"{video_name}_frame{idx}.jpg")
                    cv2.imwrite(output_path, face)
                    total_frames += 1
        
        cap.release()
    
    # Process FAKE videos (all manipulation folders)
    fake_folders = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    for folder in fake_folders:
        fake_videos = glob.glob(os.path.join(DATASET_PATH, folder, '**', '*.mp4'), recursive=True)
        print(f"\n📹 Found {len(fake_videos)} FAKE videos from {folder}")
        
        for video_path in tqdm(fake_videos, desc=f"Processing {folder}"):
            video_count += 1
            video_name = os.path.basename(video_path).replace('.mp4', '')
            
            cap = cv2.VideoCapture(video_path)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_indices = np.linspace(0, total_frames_video-1, FRAMES_PER_VIDEO, dtype=int)
            
            for idx, frame_num in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    face = detect_face(frame)
                    
                    if face is not None:
                        output_path = os.path.join(OUTPUT_PATH, 'fake', f"{folder}_{video_name}_frame{idx}.jpg")
                        cv2.imwrite(output_path, face)
                        total_frames += 1
            
            cap.release()
    
    print(f"\n✅ Extracted {total_frames} faces from {video_count} videos")
    print(f"   REAL faces: {len(os.listdir(os.path.join(OUTPUT_PATH, 'real')))}")
    print(f"   FAKE faces: {len(os.listdir(os.path.join(OUTPUT_PATH, 'fake')))}")

if __name__ == "__main__":
    import numpy as np
    extract_frames_from_videos()
