# Ear - Next Session

## Vocal Type Detection (main issue)

Currently calling everyone "male (low)" or "androgynous" even when obviously wrong (Billie Eilish detected as bass/baritone).

**Root cause:** Pitch detection picking up bass/instrumental frequencies despite 120Hz high-pass filter.

**Potential fixes:**
- Raise high-pass cutoff to 200Hz+
- More aggressive magnitude filtering - only trust strong pitch detections
- Weight toward highest consistent pitches in vocal range (vocals usually highest melodic element)
- Consider using a proper vocal activity detector to isolate frames with clear vocals

**Test cases:**
- `billie.mp3` - should detect female, currently says male (low)
- `Blinded.mp3` - should detect male tenor/falsetto, currently says male (low)

## Whisper Transcription (minor)

Mostly works great. Two edge cases:
- `billie.mp3` - hallucinated "My Invisalign has-" repeated garbage
- `Blinded.mp3` - returned nothing but musical note symbols

Not a priority - occasional mishearing is fine, matches human experience.

---

*Notes from 2026-02-19 session review*
