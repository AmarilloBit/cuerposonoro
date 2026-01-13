"""
OSC sender for communicating with SuperCollider.
"""

from pythonosc import udp_client


class OSCSender:
    """Sends motion features to SuperCollider via OSC."""

    def __init__(self, host: str = "127.0.0.1", port: int = 57120):
        """
        Initialize OSC client.

        Args:
            host: SuperCollider host (default localhost)
            port: SuperCollider OSC port (default 57120)
        """
        self.host = host
        self.port = port
        self.client = udp_client.SimpleUDPClient(host, port)
        print(f"OSC sender ready: {host}:{port}")

    def send_features(self, features: dict):
        """
        Send all features as individual OSC messages.

        Args:
            features: dict with energy, symmetry, smoothness, etc.
        """
        for name, value in features.items():
            address = f"/motion/{name}"
            self.client.send_message(address, float(value))

    def send(self, address: str, value: float):
        """
        Send a single OSC message.

        Args:
            address: OSC address (e.g., "/motion/energy")
            value: Float value to send
        """
        self.client.send_message(address, float(value))

    def send_bundle(self, features: dict):
        """
        Send all features as a single OSC bundle (lower latency).

        Args:
            features: dict with all feature values
        """
        from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
        from pythonosc.osc_message_builder import OscMessageBuilder

        bundle_builder = OscBundleBuilder(IMMEDIATELY)

        for name, value in features.items():
            msg_builder = OscMessageBuilder(address=f"/motion/{name}")
            msg_builder.add_arg(float(value))
            bundle_builder.add_content(msg_builder.build())

        bundle = bundle_builder.build()
        self.client.send(bundle)
