import cv2
import mediapipe as mp

# Inisialisasi MediaPipe untuk tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

# Fungsi untuk menghitung jari yang terangkat
def count_fingers(hand_landmarks):
    # Definisikan indeks landmark untuk jari: 0 adalah pergelangan, 4 adalah jempol, dst.
    finger_tips = [4, 8, 12, 16, 20]  # Landmark ujung jari
    finger_bases = [3, 6, 10, 14, 18] # Landmark pangkal jari
    fingers_up = 0

    # Loop untuk mengecek setiap jari (kecuali jempol yang dihitung berbeda)
    for i in range(1, 5):  # Lewati jempol, cek dari telunjuk sampai kelingking
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_bases[i]].y:
            fingers_up += 1

    # Cek jempol secara terpisah, dengan membandingkan posisi horizontal (x)
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_bases[0]].x:
        fingers_up += 1

    return fingers_up

while True:
    success, img = cap.read()  # Membaca frame dari kamera
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB
    result = hands.process(img_rgb)  # Proses deteksi tangan

    # Jika terdeteksi tangan
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)  # Gambar garis tangan

            # Hitung jumlah jari yang terangkat
            fingers_up = count_fingers(hand_lms)

            # Tampilkan jumlah jari yang terangkat pada gambar
            cv2.putText(img, f"Jari Terangkat: {fingers_up}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan gambar
    cv2.imshow("Deteksi Jari", img)

    # Keluar jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepas kamera dan menutup semua jendela
cap.release()
cv2.destroyAllWindows()
