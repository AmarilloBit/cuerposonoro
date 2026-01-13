// Cuerpo Sonoro - SuperCollider Audio Engine
// Run this before starting the Python integration test

// 1. Boot server (if not already running)
(
s.options.numInputBusChannels = 0;
s.options.outDevice = "WH-1000XM4";  // Change to your output device
s.boot;
)

// 2. Wait for server to boot, then run this block
(
// SynthDef controlled by body motion
SynthDef(\bodySound, {
    arg freq=220, amp=0.3, pan=0,
        cutoff=2000, resonance=0.3,
        attack=0.1, release=0.5;

    var sig, env;

    // Oscillator with slight detuning for richness
    sig = LFSaw.ar(freq) + LFSaw.ar(freq * 1.01);
    sig = sig * 0.5;

    // Filter controlled by motion
    sig = RLPF.ar(sig, cutoff, resonance);

    // Amplitude envelope
    env = EnvGen.kr(Env.asr(attack, 1, release), doneAction: 0);
    sig = sig * env * amp;

    // Stereo panning
    sig = Pan2.ar(sig, pan);

    Out.ar(0, sig);
}).add;

// Create synth instance
~bodySynth = Synth(\bodySound);

// OSC listeners for motion features
OSCdef(\energy, { |msg|
    var value = msg[1];
    // Energy controls filter cutoff and amplitude
    ~bodySynth.set(\cutoff, value.linexp(0, 1, 200, 4000));
    ~bodySynth.set(\amp, value.linlin(0, 1, 0.1, 0.5));
    ("Energy: " ++ value.round(0.01)).postln;
}, '/motion/energy');

OSCdef(\symmetry, { |msg|
    var value = msg[1];
    // Symmetry controls stereo panning
    ~bodySynth.set(\pan, value);
    ("Symmetry: " ++ value.round(0.01)).postln;
}, '/motion/symmetry');

OSCdef(\armAngle, { |msg|
    var value = msg[1];
    // Arm angle controls pitch
    ~bodySynth.set(\freq, value.linexp(0, 1, 110, 440));
    ("Arm Angle: " ++ value.round(0.01)).postln;
}, '/motion/armAngle');

OSCdef(\smoothness, { |msg|
    var value = msg[1];
    // Smoothness controls resonance
    ~bodySynth.set(\resonance, value.linlin(0, 1, 0.1, 0.5));
}, '/motion/smoothness');

OSCdef(\verticalExtension, { |msg|
    var value = msg[1];
    // Vertical extension adds harmonic content
    // (optional, for future use)
}, '/motion/verticalExtension');

"Cuerpo Sonoro ready! Listening for OSC on port 57120...".postln;
)

// 3. To stop everything
// Cmd + .
s.quit;
