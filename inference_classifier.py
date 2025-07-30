import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import paho.mqtt.client as mqtt

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

labels_dict = {
    0: 'me', 1: 'sorry', 2: 'thank', 3: 'hello', 4: 'introduce', 5: 'fine',
    6: 'meet', 7: 'signname', 8: 'noproblem', 9: 'unwell', 10: 'yes',11:'no'
}

current_class = None
class_start_time = None
last_detected_time = time.time()
detected_class = None
last_sent_time = None
last_sent_class = None

MQTT_BROKER = "130.33.96.46"
MQTT_PORT = 1883
MQTT_TOPIC = "mqtt/answer"

try:
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("MQTT connected successfully")
except Exception as e:
    print(f"MQTT connection failed: {e}")
    mqtt_client = None

last_detected_class = None

def process_frame(frame, detected_class, current_class, class_start_time):
    global last_detected_time, last_detected_class, last_sent_time, last_sent_class
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        all_x = []
        all_y = []
        data_aux = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            for landmark in hand_landmarks.landmark:
                all_x.append(landmark.x)
                all_y.append(landmark.y)
        
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(all_x))
                data_aux.append(landmark.y - min(all_y))
        
      
        num_hands = len(results.multi_hand_landmarks)
        if num_hands == 1:
            data_aux.extend([0] * 42)  
        
        if len(data_aux) == 84:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_class_id = int(prediction[0])
                
                if predicted_class_id in labels_dict:
                    predicted_character = labels_dict[predicted_class_id]
                    
                    if current_class == predicted_character:
                        if class_start_time is None:
                            class_start_time = time.time()
                        elif time.time() - class_start_time >= 2:
                            current_time = time.time()
                            if (last_sent_time is None or 
                                current_time - last_sent_time >= 3 or 
                                last_sent_class != predicted_character):
                                
                                detected_class = predicted_character
                                last_detected_class = predicted_character
                                
                            
                                if mqtt_client:
                                    try:
                                        mqtt_client.publish(MQTT_TOPIC, predicted_character)
                                      
                                        last_sent_time = current_time
                                        last_sent_class = predicted_character
                                    except Exception as e:
                                        print(f"MQTT publish error: {e}")
                                
                                class_start_time = current_time
                    else:
                        current_class = predicted_character
                        class_start_time = time.time()

                    last_detected_time = time.time()

                    x1, y1 = int(min(all_x) * W) - 10, int(min(all_y) * H) - 10
                    x2, y2 = int(max(all_x) * W) + 10, int(max(all_y) * H) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    if current_class == predicted_character and class_start_time:
                        time_remaining = 2 - (time.time() - class_start_time)
                        if time_remaining > 0:
                            cv2.putText(frame, f"Hold: {time_remaining:.1f}s", (x1, y2 + 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"Prediction error: {e}")
    else:
        last_detected_class = None
        current_class = None
        class_start_time = None

    return frame, detected_class, current_class, class_start_time

def draw_results(frame, detected_class):
    H, W, _ = frame.shape
    detected_text = f"Last Sent: {detected_class if detected_class else 'None'}"
    
    cv2.rectangle(frame, (10, H - 80), (W - 10, H - 10), (255, 255, 255), -1)
    cv2.putText(frame, detected_text, (20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
   
    
    return frame

def reset_detected_class():
    global detected_class
    detected_class = None

def main():
    global current_class, class_start_time, last_detected_time, detected_class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame, detected_class, current_class, class_start_time = process_frame(
            frame, detected_class, current_class, class_start_time)

        frame = draw_results(frame, detected_class)
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.disconnect()
    # print("👋 Application closed")

if __name__ == "__main__":
    main()