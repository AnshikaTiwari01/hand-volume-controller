import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# -----------------------------
# Audio Setup (PyCaw)
# -----------------------------
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

device = AudioUtilities.GetSpeakers()
interface = device.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]



# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    if not success:
      continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Get landmark positions
        lm_list = []
        h, w, _ = img.shape

        for id, lm in enumerate(hand.landmark):
            lm_list.append([id, int(lm.x * w), int(lm.y * h)])

        # Finger IDs (as per MediaPipe)
        index_tip = lm_list[8]     # index finger tip
        middle_tip = lm_list[12]   # middle finger tip

        x1, y1 = index_tip[1], index_tip[2]
        x2, y2 = middle_tip[1], middle_tip[2]

        # Draw points
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        # Draw line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Distance between fingers
        distance = math.hypot(x2 - x1, y2 - y1)

        # Map distance → volume
        vol = np.interp(distance, [20, 200], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        # Volume bar
        vol_bar = np.interp(distance, [20, 200], [400, 150])
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

        # Draw everything on the frame
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Volume Control", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
