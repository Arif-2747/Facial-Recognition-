import cv2
import numpy as np
import os
import pickle
import face_recognition # pyright: ignore[reportMissingImports]
from datetime import datetime

class RealFaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection (backup method)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load pre-trained models for age and gender detection
        self.age_net = None
        self.gender_net = None
        
        # Age and gender model paths
        self.age_model_path = "age_net.caffemodel"
        self.age_proto_path = "age_deploy.prototxt"
        self.gender_model_path = "gender_net.caffemodel"
        self.gender_proto_path = "gender_deploy.prototxt"
        
        # Age groups and gender labels
        self.age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_labels = ['Male', 'Female']
        
        # Try to load the models
        self.load_models()
        
        # Face recognition variables - NOW USING FACE ENCODINGS
        self.known_face_encodings = []  # Mathematical representations instead of raw images
        self.known_names = []
        
        # Database file path for persistence
        self.database_file = "face_encodings_database.pkl"
        
        # Recognition settings
        self.recognition_tolerance = 0.6  # Lower = more strict, Higher = more lenient
        
        # Load existing faces from file
        self.load_faces_from_file()
        
    def load_models(self):
        """Load age and gender detection models if available"""
        try:
            if os.path.exists(self.age_proto_path) and os.path.exists(self.age_model_path):
                self.age_net = cv2.dnn.readNet(self.age_model_path, self.age_proto_path)
                print("✅ Age detection model loaded successfully")
            else:
                print("⚠️ Age detection models not found. Using face detection only.")
                
            if os.path.exists(self.gender_proto_path) and os.path.exists(self.gender_model_path):
                self.gender_net = cv2.dnn.readNet(self.gender_model_path, self.gender_proto_path)
                print("✅ Gender detection model loaded successfully")
            else:
                print("⚠️ Gender detection models not found. Using face detection only.")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Running with basic face detection only")
    
    def load_faces_from_file(self):
        """Load previously saved face encodings from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_names = data['names']
                print(f"✅ Loaded {len(self.known_face_encodings)} face encodings from database")
            else:
                print("💾 No existing face database found. Starting fresh.")
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            print("💡 Starting with empty database")
            self.known_face_encodings = []
            self.known_names = []
    
    def save_faces_to_file(self):
        """Save current face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_names
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved {len(self.known_face_encodings)} face encodings to database")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def detect_age_gender(self, face_img):
        """Detect age and gender from face image"""
        age_pred = "Unknown"
        gender_pred = "Unknown"
        confidence_age = 0
        confidence_gender = 0
        
        try:
            # Prepare image for models
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Age prediction
            if self.age_net is not None:
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age_idx = np.argmax(age_preds[0])
                age_pred = self.age_groups[age_idx]
                confidence_age = age_preds[0][age_idx] * 100
            
            # Gender prediction
            if self.gender_net is not None:
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender_idx = np.argmax(gender_preds[0])
                gender_pred = self.gender_labels[gender_idx]
                confidence_gender = gender_preds[0][gender_idx] * 100
                
        except Exception as e:
            print(f"Error in age/gender detection: {e}")
            
        return age_pred, gender_pred, confidence_age, confidence_gender
    
    def add_known_face(self, frame, face_location, name):
        """Add a known face using face encoding (REAL RECOGNITION)"""
        try:
            # Extract face encoding from the detected face
            face_encodings = face_recognition.face_encodings(frame, [face_location])
            
            if face_encodings:
                face_encoding = face_encodings[0]
                
                # Check if this person already exists
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.recognition_tolerance)
                    if any(matches):
                        existing_index = matches.index(True)
                        existing_name = self.known_names[existing_index]
                        print(f"⚠️ This person already exists as '{existing_name}'")
                        
                        # Ask if user wants to update
                        update = input(f"Update '{existing_name}' to '{name}'? (y/n): ")
                        if update.lower() == 'y':
                            self.known_names[existing_index] = name
                            print(f"✅ Updated {existing_name} to {name}")
                            self.save_faces_to_file()
                        return
                
                # Add new person
                self.known_face_encodings.append(face_encoding)
                self.known_names.append(name)
                print(f"✅ Added {name} to face recognition database")
                
                # Automatically save to file
                self.save_faces_to_file()
            else:
                print("❌ Could not generate face encoding for this face")
                
        except Exception as e:
            print(f"❌ Error adding face: {e}")
    
    def recognize_faces(self, frame):
        """Recognize faces using face_recognition library (REAL RECOGNITION)"""
        recognized_faces = []
        
        try:
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Loop through each face found in the frame
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # See if the face is a match for any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.recognition_tolerance)
                name = "Unknown Person"
                confidence = 0
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        # Convert distance to confidence percentage (lower distance = higher confidence)
                        confidence = (1 - face_distances[best_match_index]) * 100
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_location,
                    'encoding': face_encoding
                })
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            
        return recognized_faces
    
    def remove_known_face(self, name):
        """Remove a face from the known faces database"""
        try:
            index = self.known_names.index(name)
            removed_name = self.known_names.pop(index)
            self.known_face_encodings.pop(index)
            print(f"🗑️ Removed {removed_name} from database")
            
            # Save changes to file
            self.save_faces_to_file()
            return True
        except ValueError:
            print(f"❌ Person '{name}' not found in database")
            return False
    
    def list_known_faces(self):
        """Display all known faces in the database"""
        if not self.known_names:
            print("📭 No faces in database")
        else:
            print(f"📋 Known faces ({len(self.known_names)}):")
            for i, name in enumerate(self.known_names):
                print(f"   {i+1}. {name}")
    
    def draw_face_info(self, frame, face_info, age=None, gender=None, conf_age=0, conf_gender=0):
        """Draw information box around detected face"""
        top, right, bottom, left = face_info['location']
        name = face_info['name']
        confidence = face_info['confidence']
        
        # Draw face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Info background
        info_height = 100
        cv2.rectangle(frame, (left, top - info_height), (right, top), (0, 0, 0), -1)
        cv2.rectangle(frame, (left, top - info_height), (right, top), (0, 255, 0), 2)
        
        # Text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Name and recognition confidence
        cv2.putText(frame, f"Name: {name}", (left + 5, top - 80), font, font_scale, color, thickness)
        if confidence > 0:
            cv2.putText(frame, f"Match: {confidence:.1f}%", (left + 5, top - 65), font, font_scale, color, thickness)
        
        # Age
        if age and conf_age > 0:
            cv2.putText(frame, f"Age: {age} ({conf_age:.1f}%)", (left + 5, top - 50), font, font_scale, color, thickness)
        elif age:
            cv2.putText(frame, f"Age: {age}", (left + 5, top - 50), font, font_scale, color, thickness)
        
        # Gender
        if gender and conf_gender > 0:
            cv2.putText(frame, f"Gender: {gender} ({conf_gender:.1f}%)", (left + 5, top - 35), font, font_scale, color, thickness)
        elif gender:
            cv2.putText(frame, f"Gender: {gender}", (left + 5, top - 35), font, font_scale, color, thickness)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (left + 5, top - 20), font, font_scale, color, thickness)
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎥 Camera started successfully!")
        print("📋 Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to save current frame")
        print("   • Press 'a' to add current face to known faces")
        print("   • Press 'c' to clear known faces")
        print("   • Press 'l' to list all known faces")
        print("   • Press 'r' to remove a face from database")
        print("   • Press 't' to adjust recognition tolerance")
        
        frame_count = 0
        process_every_n_frames = 5  # Process every 5th frame for better performance
        last_recognized_faces = []  # Cache last results
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Only process face recognition every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    
                    # REAL FACE RECOGNITION on smaller frame
                    recognized_faces = self.recognize_faces(small_frame)
                    
                    # Scale back up face locations
                    for face_info in recognized_faces:
                        top, right, bottom, left = face_info['location']
                        face_info['location'] = (top * 2, right * 2, bottom * 2, left * 2)
                    
                    # Cache results
                    last_recognized_faces = recognized_faces
                else:
                    # Use cached results for smooth display
                    recognized_faces = last_recognized_faces
                # Process and draw face information
                for face_info in recognized_faces:
                    # Extract face region for age/gender detection
                    top, right, bottom, left = face_info['location']
                    face_img = frame[top:bottom, left:right]
                    
                    if face_img.size > 0:
                        # Age and gender detection (only on processing frames)
                        if frame_count % process_every_n_frames == 0:
                            age, gender, conf_age, conf_gender = self.detect_age_gender(face_img)
                            # Cache age/gender results
                            face_info['age'] = age
                            face_info['gender'] = gender
                            face_info['conf_age'] = conf_age
                            face_info['conf_gender'] = conf_gender
                        else:
                            # Use cached age/gender results
                            age = face_info.get('age', 'Unknown')
                            gender = face_info.get('gender', 'Unknown')
                            conf_age = face_info.get('conf_age', 0)
                            conf_gender = face_info.get('conf_gender', 0)
                        
                        # Draw information
                        self.draw_face_info(frame, face_info, age, gender, conf_age, conf_gender)
                info_y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (0, 255, 0)
                thickness = 2
                small_font_scale = 0.5
                
                # System info
                if frame_count % process_every_n_frames == 0:
                    current_faces = len(recognized_faces) if 'recognized_faces' in locals() else 0
                else:
                    current_faces = 0
                    
                cv2.putText(frame, f"Faces Detected: {current_faces}", (10, info_y), 
                           font, font_scale, color, thickness)
                cv2.putText(frame, f"Known Faces: {len(self.known_face_encodings)}", (10, info_y + 30), 
                           font, font_scale, color, thickness)
                
                # Controls display
                controls_start_y = info_y + 80
                controls_color = (255, 255, 0)
                
                cv2.putText(frame, "CONTROLS:", (10, controls_start_y), 
                           font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "Q-Quit ", (10, controls_start_y + 20), 
                           font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "A - Add", (10, controls_start_y + 40),
                            font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "S - Save", (10, controls_start_y + 60),
                            font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "L - List", (10, controls_start_y + 80),
                            font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "R - Remove", (10, controls_start_y + 100),
                            font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "C-Clear", (10, controls_start_y + 120), 
                           font, small_font_scale, controls_color, 1)
                cv2.putText(frame, "T-Tolerance", (10, controls_start_y + 140),
                            font, small_font_scale, controls_color, 1)
                
                # Model status
                model_status_y = frame.shape[0] - 100
                status_color = (0, 255, 255)
                
                age_status = "OK" if self.age_net is not None else "NO"
                gender_status = "OK" if self.gender_net is not None else "NO"
                
                cv2.putText(frame, f"Age Model: {age_status}", (10, model_status_y), 
                           font, small_font_scale, status_color, 1)
                cv2.putText(frame, f"Gender Model: {gender_status}", (10, model_status_y + 20), 
                           font, small_font_scale, status_color, 1)
                cv2.putText(frame, f"Recognition: REAL", (10, model_status_y + 40), 
                           font, small_font_scale, status_color, 1)
                cv2.putText(frame, f"Tolerance: {self.recognition_tolerance}", (10, model_status_y + 60), 
                           font, small_font_scale, status_color, 1)
                
                # Current time
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Time: {current_time}", (10, model_status_y + 80), 
                           font, small_font_scale, status_color, 1)
                
                # Show frame
                cv2.imshow('Real Face Recognition System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved as {filename}")
                elif key == ord('a'):
                    # Add face - use the most recent recognition results
                    if 'recognized_faces' in locals() and recognized_faces:
                        face_info = recognized_faces[0]  # Add first detected face
                        name = input("\n👤 Enter name for this person: ")
                        if name:
                            self.add_known_face(frame, face_info['location'], name)
                    else:
                        print("⚠️ No faces detected. Please ensure a face is visible.")
                elif key == ord('c'):
                    self.known_face_encodings.clear()
                    self.known_names.clear()
                    if os.path.exists(self.database_file):
                        os.remove(self.database_file)
                    print("🗑️ Cleared all known faces and deleted database")
                elif key == ord('l'):
                    print("\n" + "="*40)
                    self.list_known_faces()
                    print("="*40)
                elif key == ord('r'):
                    if self.known_names:
                        print("\n" + "="*40)
                        self.list_known_faces()
                        name_to_remove = input("\n🗑️ Enter name to remove: ")
                        if name_to_remove:
                            self.remove_known_face(name_to_remove)
                        print("="*40)
                    else:
                        print("📭 No faces to remove")
                elif key == ord('t'):
                    print(f"\n🎯 Current tolerance: {self.recognition_tolerance}")
                    print("💡 Lower = more strict, Higher = more lenient")
                    try:
                        new_tolerance = float(input("Enter new tolerance (0.1-1.0): "))
                        if 0.1 <= new_tolerance <= 1.0:
                            self.recognition_tolerance = new_tolerance
                            print(f"✅ Updated tolerance to {new_tolerance}")
                        else:
                            print("❌ Please enter a value between 0.1 and 1.0")
                    except ValueError:
                        print("❌ Invalid input")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🔄 Camera released and windows closed")

def main():
    print("🚀 Starting REAL Face Recognition System")
    print("🧠 Using Advanced Face Encodings")
    print("=" * 60)
    
    # Check if face_recognition library is installed
    try:
        import face_recognition
        print("✅ face_recognition library found")
    except ImportError:
        print("❌ face_recognition library not found!")
        print("📥 Install with: pip install face_recognition")
        print("⚠️  Note: This requires dlib and cmake to be installed first")
        return
    
    detector = RealFaceRecognitionSystem()
    detector.run_detection()

if __name__ == "__main__":
    main()