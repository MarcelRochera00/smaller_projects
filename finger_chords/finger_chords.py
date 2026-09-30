import math

import cv2
import mediapipe as mp
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

SAMPLE_RATE = 44100
LOOP_SECONDS = 2
ATTACK_MS = 180
RELEASE_MS = 600
STABLE_FRAMES = 2

FRAME_HEIGHT = 720

WRIST = 0
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_NAMES = ["indice", "corazon", "anular", "menique"]
PALM_LINES = [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
FINGER_CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

ACCENT = (120, 220, 255)
MUTED = (160, 160, 165)
TEXT = (235, 235, 240)
DARK = (20, 22, 28)
PILL_OFF = (90, 92, 100)
FONT_CANDIDATES = ("Inter-Regular.ttf", "Ubuntu-R.ttf", "DejaVuSans.ttf", "Arial.ttf")

MELODY = [("C", 72), ("E", 76), ("G", 79), ("A", 81)]

CHORDS = {
    1: ("C", (60, 64, 67)),
    2: ("G", (55, 59, 62)),
    3: ("Am", (57, 60, 64)),
    4: ("F", (53, 57, 60)),
}

# partials: amplitud de cada armonico, detunes: (desafinación en ciclos, peso),
# tremolo: (profundidad, Hz). Los Hz ENTEROS para que el bucle funcione.
TIMBRES = [
    {"name": "Suave", 
     "partials": [1.0, 0.16, 0.03],
     "detunes": ((-1, 0.35), (0, 1.0), (1, 0.35)), 
     "tremolo": (0.0, 0)},
    {"name": "Calido", 
     "partials": [1.0, 0.40, 0.14, 0.05],
     "detunes": ((-1, 0.30), (0, 1.0), (1, 0.30)), 
     "tremolo": (0.12, 3)},
    {"name": "Etereo", 
     "partials": [1.0, 0.28, 0.08],
     "detunes": ((-2, 0.45), (0, 1.0), (2, 0.45)), 
     "tremolo": (0.22, 2)},
]
CHORD_PARTIALS = [1.0, 0.42, 0.18, 0.08, 0.04]
CHORD_DETUNES = ((-1, 0.5), (0, 1.0), (1, 0.5))

mp_hands = mp.solutions.hands

class Stabilizer:
    def __init__(self, initial, frames=STABLE_FRAMES):
        self.value = initial
        self.candidate = initial
        self.frames = frames
        self.count = 0

    def update(self, observed):
        if observed == self.value:
            self.candidate, self.count = observed, 0
        elif observed == self.candidate:
            self.count += 1
            if self.count >= self.frames:
                self.value, self.count = observed, 0
        else:
            self.candidate, self.count = observed, 1
        return self.value


def midi_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def synthesize(notes, volume, partials, detunes=((0, 1.0),), tremolo=(0.0, 0)):
    t = np.arange(SAMPLE_RATE * LOOP_SECONDS) / SAMPLE_RATE
    wave = np.zeros_like(t)
    for note in notes:
        base_cycles = round(midi_to_freq(note) * LOOP_SECONDS)
        for detune, weight in detunes:
            freq = (base_cycles + detune) / LOOP_SECONDS
            for n, amp in enumerate(partials, start=1):
                wave += weight * amp * np.sin(2 * np.pi * freq * n * t)

    depth, rate = tremolo
    if depth:
        wave *= 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))

    wave = wave / np.abs(wave).max() * volume
    samples = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack([samples, samples])))


def build_melody(timbre):
    return [
        synthesize([note], 0.15, timbre["partials"], timbre["detunes"], timbre["tremolo"])
        for _, note in MELODY
    ]


def build_chords():
    return {
        count: synthesize(notes, 0.45, CHORD_PARTIALS, CHORD_DETUNES)
        for count, (_, notes) in CHORDS.items()
    }


def update_chord(sounds, previous, current):
    if previous == current:
        return
    if previous in sounds:
        sounds[previous].fadeout(RELEASE_MS)
    if current in sounds:
        sounds[current].play(loops=-1, fade_ms=ATTACK_MS)


def update_melody(sounds, previous, current):
    for sound, was_up, is_up in zip(sounds, previous, current):
        if is_up and not was_up:
            sound.play(loops=-1, fade_ms=ATTACK_MS)
        elif was_up and not is_up:
            sound.fadeout(RELEASE_MS)


def raised_fingers(hand):
    # Se compara la distancia a el wrist en vez de la altura, asi funciona con la mano girada
    lm = hand.landmark

    def dist(i):
        return math.hypot(lm[i].x - lm[WRIST].x, lm[i].y - lm[WRIST].y)

    return [dist(tip) > dist(pip) * 1.04 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]


def load_font(size):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_text(frame, items):
    image = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for text, position, font, color, *anchor in items:
        draw.text(position, text, font=font, fill=color, anchor=anchor[0] if anchor else "la")
    return np.ascontiguousarray(np.array(image)[:, :, ::-1])


def panel(frame, x1, y1, x2, y2, color=DARK, alpha=0.55, radius=14):
    w, h = x2 - x1, y2 - y1
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx in (radius, w - radius):
        for cy in (radius, h - radius):
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    roi = frame[y1:y2, x1:x2]
    tinted = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color[::-1]), alpha, 0)
    roi[mask > 0] = tinted[mask > 0]


def draw_hand(frame, hand, raised):
    height, width = frame.shape[:2]
    lm = hand.landmark
    points = {i: (int(lm[i].x * width), int(lm[i].y * height)) for i in range(21)}
    idle = (120, 120, 125)

    for a, b in PALM_LINES:
        cv2.line(frame, points[a], points[b], idle, 2, cv2.LINE_AA)

    for chain, up in zip(FINGER_CHAINS, raised):
        line_color = ACCENT[::-1] if up else idle
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, points[a], points[b], line_color, 3 if up else 2, cv2.LINE_AA)
        for i in chain:
            cv2.circle(frame, points[i], 3, line_color, -1, cv2.LINE_AA)

    cv2.circle(frame, points[WRIST], 4, (200, 200, 205), -1, cv2.LINE_AA)


def draw_chord_card(frame, labels, fonts, chord):
    x1, y1, x2, y2 = 24, 20, 324, 172
    panel(frame, x1, y1, x2, y2, radius=16)

    name = CHORDS[chord][0] if chord in CHORDS else "-"
    labels.append(("ACORDE · MANO IZQUIERDA", (x1 + 18, y1 + 14), fonts["caption"], MUTED))
    labels.append((name, (x1 + 18, y1 + 26), fonts["chord"], ACCENT if chord else MUTED))

    for number, (chord_name, _) in CHORDS.items():
        x = x1 + 18 + (number - 1) * 70
        y = y2 - 50
        active = number == chord
        panel(frame, x, y, x + 60, y + 32, ACCENT if active else PILL_OFF,
              0.92 if active else 0.45, radius=10)
        labels.append((f"{number} · {chord_name}", (x + 30, y + 16), fonts["small"],
                       DARK if active else MUTED, "mm"))


def draw_timbre_card(frame, labels, fonts, timbre_name):
    x2 = frame.shape[1] - 24
    x1, y1, y2 = x2 - 270, 20, 112
    panel(frame, x1, y1, x2, y2, radius=16)
    labels.append(("TIMBRE · MANO DERECHA", (x1 + 18, y1 + 14), fonts["caption"], MUTED))
    labels.append((timbre_name, (x1 + 18, y1 + 32), fonts["title"], TEXT))
    labels.append(("T  cambiar      Q  salir", (x1 + 18, y1 + 68), fonts["caption"], MUTED))


def draw_note_bar(frame, labels, fonts, melody):
    frame_height, frame_width = frame.shape[:2]
    pill_w, pill_h, gap = 120, 64, 16
    total = len(MELODY) * pill_w + (len(MELODY) - 1) * gap
    left = (frame_width - total) // 2
    y = frame_height - 26 - pill_h

    for i, (note_name, _) in enumerate(MELODY):
        x = left + i * (pill_w + gap)
        active = melody[i]
        panel(frame, x, y, x + pill_w, y + pill_h, ACCENT if active else PILL_OFF,
              0.92 if active else 0.45)
        labels.append((note_name, (x + pill_w // 2, y + 24), fonts["note"],
                       DARK if active else TEXT, "mm"))
        labels.append((FINGER_NAMES[i], (x + pill_w // 2, y + 48), fonts["caption"],
                       DARK if active else MUTED, "mm"))


def draw_fingertips(frame, tips, melody):
    for (x, y), is_up in zip(tips, melody):
        if is_up:
            cv2.circle(frame, (x, y), 18, ACCENT[::-1], 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, ACCENT[::-1], -1, cv2.LINE_AA)


def main():
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
    pygame.mixer.set_num_channels(16)

    timbre_index = 0
    melody_sounds = build_melody(TIMBRES[timbre_index])
    chord_sounds = build_chords()

    fonts = {
        "caption": load_font(12),
        "small": load_font(14),
        "title": load_font(26),
        "chord": load_font(64),
        "note": load_font(28),
        "hint": load_font(20),
    }

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0)

    chord_filter = Stabilizer(0)
    melody_filters = [Stabilizer(False) for _ in MELODY]
    previous_chord = 0
    previous_melody = [False] * len(MELODY)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # mantener proporcion original de la camara
        frame = cv2.flip(frame, 1)
        scale = FRAME_HEIGHT / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), FRAME_HEIGHT))
        frame = cv2.convertScaleAbs(frame, alpha=0.55)
        height, width = frame.shape[:2]
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        chord = 0
        melody = [False] * len(MELODY)
        tips = []

        if result.multi_hand_landmarks:
            for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                fingers = raised_fingers(hand)
                draw_hand(frame, hand, fingers)
                if handedness.classification[0].label == "Left":
                    chord = sum(fingers)
                else:
                    melody = fingers
                    tips = [(int(hand.landmark[t].x * width), int(hand.landmark[t].y * height))
                            for t in FINGER_TIPS]

        chord = chord_filter.update(chord)
        melody = [f.update(value) for f, value in zip(melody_filters, melody)]
        update_chord(chord_sounds, previous_chord, chord)
        update_melody(melody_sounds, previous_melody, melody)
        previous_chord, previous_melody = chord, melody

        labels = []
        draw_chord_card(frame, labels, fonts, chord)
        draw_timbre_card(frame, labels, fonts, TIMBRES[timbre_index]["name"])
        draw_note_bar(frame, labels, fonts, melody)
        draw_fingertips(frame, tips, melody)

        if not result.multi_hand_landmarks:
            labels.append(("Muestra tus manos", (width // 2, height // 2 - 14),
                           fonts["title"], TEXT, "mm"))
            labels.append(("izquierda = acordes   ·   derecha = melodia",
                           (width // 2, height // 2 + 22), fonts["hint"], MUTED, "mm"))

        cv2.imshow("Finger Synth", render_text(frame, labels))

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            for sound in melody_sounds:
                sound.fadeout(150)
            timbre_index = (timbre_index + 1) % len(TIMBRES)
            melody_sounds = build_melody(TIMBRES[timbre_index])
            melody_filters = [Stabilizer(False) for _ in MELODY]
            previous_melody = [False] * len(MELODY)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()