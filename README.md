# ImageToAudio: Bilder mit Sound übertragen

Eine Python-Anwendung, die es ermöglicht, Bilder in Audiosignale umzuwandeln und mithilfe von **Frequency Shift Keying (FSK)** zu übertragen. Die Audiosignale können über Lautsprecher abgespielt und mit einem Mikrofon empfangen werden, um das Bild auf der Empfängerseite zu rekonstruieren.

---

## 🚀 Funktionsübersicht

1. **Bild zu Audiosignal:**
   - Ein Bild wird in Binärdaten umgewandelt.
   - Die Binärdaten werden mit FSK kodiert, wobei Frequenzen verschiedene Daten repräsentieren.
   - Das resultierende Signal wird in einer WAV-Datei gespeichert.

2. **Audiosignal zu Bild:**
   - Ein aufgenommenes Audiosignal wird analysiert.
   - Die kodierten Binärdaten werden dekodiert und das Bild daraus
