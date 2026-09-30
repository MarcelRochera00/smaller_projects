# Finger Chords

A small real-time synth you play with your hands in front of a webcam. Your **left hand plays chords** and your **right hand plays melody notes**, depending on which fingers you raise.

<!-- Add a screenshot or a short GIF here: ![Demo](demo.gif) -->

## How it works

1. **OpenCV** reads your webcam and then draws the interface (mirrored so you can see it correctly).
2. **MediaPipe Hands** (a pretrained hand-tracking model) gets 21 landmarks per hand and tells left from right.
3. I counted a finger as raised when its tip was farther from the wrist than its middle joint, so it still works if your hand is rotated. The code also waits for 2 consecutive frames before changing anything, which avoids flickering notes and weird sounds.
4. The sound is **synthesized from scratch with NumPy** (harmonics, slight detuning and tremolo) and played with **pygame**. Notes fade in and out so they don't click and you get a clean sound.

## Controls

**Left hand: number of raised fingers = chord**

| Fingers | Chord |
|---------|-------|
| 1       | C     |
| 2       | G     |
| 3       | Am    |
| 4       | F     |

**Right hand: each finger = one note**

| Finger | Note |
|--------|------|
| Index  | C    |
| Middle | E    |
| Ring   | G    |
| Pinky  | A    |

The thumb is not used. Press **T** to cycle through three timbres (Suave o Calido) and **Q** to quit. You can change the notes in the code to generate other melodies and sounds or even add the thumb. I removed it because it was hard to detect and it did some flickering.

## Requirements

- Python 3.x and a webcam
- `opencv-python`, `mediapipe`, `numpy`, `pygame`, `pillow`

```bash
pip install -r requirements.txt
```

## Run it

```bash
python finger_chords.py
```

## Credits

Idea inspired by a TikTok video 

## Ideas for improving it

- More chords, or a way to switch keys
- Adding facial recognition with MediaPipe to "sing" notes out of your mouth without actually saying anything. Could be cool to implement and use.
- Use the thumb, or detect hand height to control volume
- Different instrument sounds