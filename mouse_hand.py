import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import os
import time

# غیرفعال کردن هشدارها
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# تنظیمات pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# صفحه نمایش
screen_width, screen_height = pyautogui.size()

# مقیاس حرکت
MOVEMENT_SCALE = 4

# Smoothing
smooth_factor = 0.35
prev_x, prev_y = pyautogui.position()

# وضعیت کلیک‌ها
left_click_cooldown = 0
right_click_cooldown = 0
COOLDOWN_FRAMES = 8

# ✨ وضعیت‌های جدید
is_dragging = False  # وضعیت درگ
scroll_cooldown = 0  # کول‌داون اسکرول
last_scroll_y = None  # آخرین موقعیت اسکرول

# FPS
fps_counter = 0
fps_start_time = time.time()
current_fps = 0


def calculate_distance(point1, point2):
    """محاسبه فاصله بین دو نقطه"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_palm_center(hand_landmarks):
    """گرفتن مرکز کف دست"""
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]

    center_x = (wrist.x + middle_mcp.x) / 2
    center_y = (wrist.y + middle_mcp.y) / 2

    return center_x, center_y


def get_finger_status(hand_landmarks):
    """تشخیص وضعیت انگشتان با تحمل چرخش دست"""
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    fingers_open = []

    # انگشت شست
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]

    thumb_dist = calculate_distance(
        [thumb_tip.x, thumb_tip.y], [wrist.x, wrist.y])
    thumb_base_dist = calculate_distance(
        [thumb_ip.x, thumb_ip.y], [wrist.x, wrist.y])
    fingers_open.append(1 if thumb_dist > thumb_base_dist * 1.1 else 0)

    # چهار انگشت دیگر
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tips[i]]
        pip = hand_landmarks.landmark[pips[i]]
        mcp = hand_landmarks.landmark[pips[i] - 1]

        finger_length = calculate_distance([tip.x, tip.y], [mcp.x, mcp.y])
        pip_distance = calculate_distance([pip.x, pip.y], [mcp.x, mcp.y])

        fingers_open.append(1 if finger_length > pip_distance * 1.15 else 0)

    return fingers_open


def detect_gesture(hand_landmarks):
    """تشخیص ژست‌های دست راست"""
    fingers = get_finger_status(hand_landmarks)

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]    # سبابه
    middle_tip = hand_landmarks.landmark[12]   # وسط
    ring_tip = hand_landmarks.landmark[16]     # انگشتری
    pinky_tip = hand_landmarks.landmark[20]    # کوچک

    # فواصل برای کلیک‌ها
    thumb_index_dist = calculate_distance(
        [thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y])
    thumb_middle_dist = calculate_distance([thumb_tip.x, thumb_tip.y], [
                                           middle_tip.x, middle_tip.y])

    # ✨ ژست اسکرول: انگشت اشاره و وسط باز (V شکل)
    if fingers == [0, 1, 1, 0, 0]:
        return "SCROLL"

    # ✨ ژست درگ: مشت بسته (همه انگشت‌ها جمع)
    if fingers == [0, 0, 0, 0, 0]:
        return "DRAG"

    # کلیک راست (وسط + شست)
    if thumb_middle_dist < 0.03 and fingers[2] == 1:
        return "RIGHT_CLICK"

    # دابل کلیک: سبابه + شست (1.5 برابر آستانه)
    if thumb_index_dist < 0.02 and fingers[1] == 1:
        return "DOUBLE_CLICK"

    # کلیک چپ (سبابه + شست) - آستانه معمولی
    if thumb_index_dist < 0.03 and fingers[1] == 1:
        return "LEFT_CLICK"

    # حرکت موس
    return "MOVE"


def map_to_screen(cam_x, cam_y, frame_width, frame_height):
    """نگاشت دوربین به مانیتور با مقیاس افزایش یافته"""
    margin = 0.08

    cam_x = np.clip(cam_x, margin, 1.0 - margin)
    cam_y = np.clip(cam_y, margin, 1.0 - margin)

    norm_x = (cam_x - margin) / (1.0 - 2 * margin)
    norm_y = (cam_y - margin) / (1.0 - 2 * margin)

    norm_x = 0.5 + (norm_x - 0.5) * MOVEMENT_SCALE
    norm_y = 0.5 + (norm_y - 0.5) * MOVEMENT_SCALE

    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_y = np.clip(norm_y, 0.0, 1.0)

    screen_x = int(norm_x * screen_width)
    screen_y = int(norm_y * screen_height)

    return screen_x, screen_y


# باز کردن دوربین
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=" * 60)
print("🎯 کنترل موس حرفه‌ای با دست - نسخه کامل")
print("=" * 60)
print(f"📐 مقیاس حرکت: {MOVEMENT_SCALE}x")
print("✋  کف دست باز: جابجایی موس")
print("👆  سبابه + شست: کلیک چپ")
print("🖕  وسط + شست: کلیک راست")
print("👌  سبابه + شست (محکم): دابل کلیک")
print("✊  مشت بسته: درگ کردن (Drag & Drop)")
print("✌️  V (سبابه + وسط): اسکرول عمودی")
print("⌨️  کلید 'q': خروج")
print("=" * 60)

# گرم کردن دوربین
for _ in range(10):
    cap.read()

frame_count = 0
SCROLL_SENSITIVITY = 40  # حساسیت اسکرول

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # کاهش رزولوشن برای سرعت
    process_frame = cv2.resize(frame, (320, 240))
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # کاهش cooldown
    if left_click_cooldown > 0:
        left_click_cooldown -= 1
    if right_click_cooldown > 0:
        right_click_cooldown -= 1
    if scroll_cooldown > 0:
        scroll_cooldown -= 1

    current_gesture = "MOVE"
    palm_x, palm_y = 0.5, 0.5

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # رسم landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2)
            )

            # مرکز کف دست
            palm_x, palm_y = get_palm_center(hand_landmarks)
            palm_x_frame = int(palm_x * frame_width)
            palm_y_frame = int(palm_y * frame_height)

            # تشخیص ژست
            current_gesture = detect_gesture(hand_landmarks)

            # ✨ رنگ نقطه مرکزی بر اساس ژست
            gesture_colors = {
                "MOVE": (0, 255, 0),        # سبز
                "LEFT_CLICK": (0, 255, 255),  # زرد
                "RIGHT_CLICK": (255, 100, 255),  # بنفش
                "DOUBLE_CLICK": (255, 255, 0),  # طلایی
                "DRAG": (255, 0, 0),         # قرمز
                "SCROLL": (0, 150, 255)       # آبی
            }
            color = gesture_colors.get(current_gesture, (255, 255, 255))

            cv2.circle(frame, (palm_x_frame, palm_y_frame), 15, color, -1)
            cv2.circle(frame, (palm_x_frame, palm_y_frame), 20, color, 3)

            # ✨ نمایش کلیک‌ها
            if current_gesture in ["LEFT_CLICK", "DOUBLE_CLICK"]:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                cv2.line(frame,
                         (int(thumb_tip.x * frame_width),
                          int(thumb_tip.y * frame_height)),
                         (int(index_tip.x * frame_width),
                          int(index_tip.y * frame_height)),
                         (0, 255, 255), 4)
                cv2.circle(frame,
                           (int(index_tip.x * frame_width),
                            int(index_tip.y * frame_height)),
                           8, (0, 255, 255), -1)

            elif current_gesture == "RIGHT_CLICK":
                thumb_tip = hand_landmarks.landmark[4]
                middle_tip = hand_landmarks.landmark[12]
                cv2.line(frame,
                         (int(thumb_tip.x * frame_width),
                          int(thumb_tip.y * frame_height)),
                         (int(middle_tip.x * frame_width),
                          int(middle_tip.y * frame_height)),
                         (255, 100, 255), 4)
                cv2.circle(frame,
                           (int(middle_tip.x * frame_width),
                            int(middle_tip.y * frame_height)),
                           8, (255, 100, 255), -1)

            # ✨ نمایش خط اسکرول
            elif current_gesture == "SCROLL":
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                cv2.line(frame,
                         (int(index_tip.x * frame_width),
                          int(index_tip.y * frame_height)),
                         (int(middle_tip.x * frame_width),
                          int(middle_tip.y * frame_height)),
                         (0, 150, 255), 4)

    # ✨ تبدیل موقعیت کف دست به مختصات موس
    mouse_x, mouse_y = map_to_screen(palm_x, palm_y, frame_width, frame_height)

    # Smoothing
    smoothed_x = prev_x + (mouse_x - prev_x) * smooth_factor
    smoothed_y = prev_y + (mouse_y - prev_y) * smooth_factor

    # ✨ حرکت موس
    pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0)
    prev_x, prev_y = smoothed_x, smoothed_y

    # ✨ اعمال کلیک‌ها و ژست‌های خاص
    if current_gesture == "LEFT_CLICK" and left_click_cooldown == 0:
        pyautogui.click()
        left_click_cooldown = COOLDOWN_FRAMES
        cv2.putText(frame, "👆 LEFT CLICK!", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.rectangle(frame, (0, 0), (frame_width,
                      frame_height), (0, 255, 255), 4)

    elif current_gesture == "DOUBLE_CLICK" and left_click_cooldown == 0:
        # ✨ دابل کلیک
        pyautogui.doubleClick()
        left_click_cooldown = COOLDOWN_FRAMES * 2
        cv2.putText(frame, "👌 DOUBLE CLICK!", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.rectangle(frame, (0, 0), (frame_width,
                      frame_height), (255, 255, 0), 4)

    elif current_gesture == "RIGHT_CLICK" and right_click_cooldown == 0:
        pyautogui.rightClick()
        right_click_cooldown = COOLDOWN_FRAMES
        cv2.putText(frame, "🖕 RIGHT CLICK!", (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 3)
        cv2.rectangle(frame, (0, 0), (frame_width, frame_height),
                      (255, 100, 255), 4)

    elif current_gesture == "DRAG":
        # ✨ درگ کردن
        if not is_dragging:
            pyautogui.mouseDown()  # نگه داشتن کلیک
            is_dragging = True
            cv2.putText(frame, "✊ DRAGGING...", (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        else:
            cv2.putText(frame, "✊ DRAGGING...", (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # خطوط قرمز چشمک‌زن دور فریم
        cv2.rectangle(frame, (5, 5), (frame_width-5,
                      frame_height-5), (255, 0, 0), 3)

    elif current_gesture == "SCROLL":
        # ✨ اسکرول کردن
        if last_scroll_y is not None and scroll_cooldown == 0:
            scroll_delta = last_scroll_y - palm_y  # جهت معکوس
            scroll_amount = int(scroll_delta * SCROLL_SENSITIVITY * 10)

            if abs(scroll_amount) > 0:
                pyautogui.scroll(scroll_amount)
                scroll_cooldown = 2

                direction = "⬆️" if scroll_amount > 0 else "⬇️"
                cv2.putText(frame, f"✌️ SCROLL {direction} {abs(scroll_amount)}", (50, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 3)

        last_scroll_y = palm_y
        # کادر آبی برای اسکرول
        cv2.rectangle(frame, (5, 5), (frame_width-5,
                      frame_height-5), (0, 150, 255), 3)

    else:
        # ✨ رها کردن درگ
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
            cv2.putText(frame, "✅ DROPPED!", (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # ریست اسکرول
        last_scroll_y = None

    # محاسبه FPS
    frame_count += 1
    if frame_count % 30 == 0:
        current_fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    # ✨ نمایش اطلاعات
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {current_fps:.1f} | Scale: {MOVEMENT_SCALE}x", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ✨ نمایش وضعیت‌های خاص
    if is_dragging:
        cv2.putText(frame, "🔴 DRAG MODE", (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # کادر محدوده فعال
    margin_px = int(0.08 * frame_width)
    cv2.rectangle(frame,
                  (margin_px, margin_px),
                  (frame_width - margin_px, frame_height - margin_px),
                  (100, 100, 100), 1)

    # ✨ راهنمای کامل
    cv2.rectangle(frame, (5, frame_height - 80), (frame_width - 5, frame_height - 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, "V=Scroll | Fist=Drag | Pinch=Click | Hard Pinch=Double | Middle+Thumb=Right",
                (15, frame_height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "Move=Move | 'q'=Exit",
                (15, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow('Pro Hand Mouse - Full Features', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✨ آزاد کردن درگ موقع خروج
if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()
print("👋 برنامه خاتمه یافت!")
