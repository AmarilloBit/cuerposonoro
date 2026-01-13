"""
Interactive MIDI test - keeps port open for Surge XT connection.

Steps:
1. Run this script
2. Open Surge XT → Settings → Audio/MIDI Settings
3. In "Active MIDI inputs" you should see "Cuerpo Sonoro" - click to enable it
4. Come back to this terminal and press Enter to continue
"""

import sys

sys.path.insert(0, '.')

from vision_processor.midi_sender import MidiSender
import time


def main():
    print("=" * 60)
    print("   INTERACTIVE MIDI TEST")
    print("=" * 60)

    print("\n1. Creating MIDI port 'Cuerpo Sonoro'...")
    sender = MidiSender()

    print("\n" + "=" * 60)
    print("   PORT IS NOW OPEN!")
    print("=" * 60)
    print("""
NOW DO THIS:
    1. Go to Surge XT
    2. Click 'Settings...' (top right)
    3. Look at 'Active MIDI inputs'
    4. You should see 'Cuerpo Sonoro' - CLICK IT to enable
    5. Come back here and press ENTER
    """)

    input("Press ENTER when Surge XT is connected...")

    print("\n" + "=" * 60)
    print("   TESTING CHORDS")
    print("=" * 60)

    features = {
        "feetCenterX": 0.1,
        "hipTilt": 0.0,
        "kneeAngle": 0.8,
        "rightHandY": 0.5,
        "leftHandY": 0.5,
        "rightHandJerk": 0.0,
        "leftHandJerk": 0.0,
        "rightArmVelocity": 0.5,
        "leftArmVelocity": 0.5,
        "rightElbowHipAngle": 0.0,
        "leftElbowHipAngle": 0.0,
        "headTilt": 0.0,
    }

    chords = [
        (0.1, "I - Do Mayor"),
        (0.35, "IV - Fa Mayor"),
        (0.6, "V - Sol Mayor"),
        (0.85, "VI - La menor"),
    ]

    for position, name in chords:
        print(f"\n→ Playing chord: {name}")
        features["feetCenterX"] = position
        sender.update(features)
        input("   Press ENTER for next chord...")

    print("\n" + "=" * 60)
    print("   TESTING MELODY")
    print("=" * 60)

    # Back to chord I
    features["feetCenterX"] = 0.1
    sender.update(features)

    print("\nTriggering melody notes...")

    for hand_y in [0.2, 0.5, 0.8]:
        print(f"\n→ Right hand at height {hand_y}")
        features["rightHandY"] = hand_y
        features["rightHandJerk"] = 0.8  # Trigger
        sender.update(features)
        features["rightHandJerk"] = 0.0  # Reset
        time.sleep(0.1)
        sender.update(features)
        input("   Press ENTER for next note...")

    print("\n" + "=" * 60)
    print("   TEST COMPLETE")
    print("=" * 60)
    print("\nPress ENTER to close the MIDI port and exit...")
    input()

    sender.close()
    print("Done!")


if __name__ == "__main__":
    main()