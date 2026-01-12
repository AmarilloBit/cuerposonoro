#!/usr/bin/env python3
"""
Test OSC communication with SuperCollider.

Before running:
1. Open SuperCollider
2. Boot server: s.boot;
3. Run the OSCdef listener (see README)

Usage:
    python tests/test_osc.py
"""

from pythonosc import udp_client
import time

# SuperCollider default port
SC_IP = "127.0.0.1"
SC_PORT = 57120


def main():
    print(f"Sending OSC to {SC_IP}:{SC_PORT}")
    print("Make sure SuperCollider is running with OSCdef listener\n")

    client = udp_client.SimpleUDPClient(SC_IP, SC_PORT)

    # Simulate changing frequency based on "energy"
    # Low energy = low freq, high energy = high freq
    frequencies = [220, 330, 440, 550, 660, 550, 440, 330, 220]

    print("Sending frequency values...")
    for freq in frequencies:
        print(f"  /motion/energy -> {freq}")
        client.send_message("/motion/energy", freq)
        time.sleep(0.5)

    print("\nDone! Check SuperCollider post window for received values.")


if __name__ == "__main__":
    main()