This folder contains the next ESP32 bridge firmware to flash for the drawing robot.

Main sketch:
- `esp32_grbl_bridge_plotter_hardened.ino`

Local credentials:
- copy `secrets.example.h` to `secrets.h`
- set `WIFI_SSID`, `WIFI_PASSWORD`, and optionally `OTA_PASSWORD`
- `secrets.h` is ignored by git and should not be committed

Why this version exists:
- reduces bridge log churn during long drawings
- trims the `/status` response so the ESP32 has less JSON work to do
- disables Wi-Fi sleep to reduce random bridge stalls
- sets the ESP32 Wi-Fi hostname to `esp32-grbl-bridge`
- adds lightweight `/health` and `/restart` endpoints for V3.0 recovery handshakes
- keeps the same HTTP API shape the Streamlit app already expects

What changed compared with the older sketch:
- rolling log reduced from 120 lines to 48
- HTTP command and realtime spam are no longer added to the log by default
- `/status` no longer includes the servo JSON blob
- `WiFi.setSleep(false)` is enabled
- `WiFi.setHostname("esp32-grbl-bridge")` is enabled
- Arduino OTA is enabled for future wireless sketch updates
- `/health` reports uptime, free heap, RSSI, IP, and the last GRBL status line
- `/restart` lets the Python backend soft-reboot the ESP32 bridge during automatic recovery

Flash intent:
- open this folder in Arduino IDE
- compile/upload this sketch to the ESP32
- after flashing, re-test the same drawing from the app

OTA notes:
- this still requires one USB flash first so the OTA-capable sketch is on the ESP32
- after that, the board should advertise itself with hostname `esp32-grbl-bridge`
- `OTA_PASSWORD` is currently blank in the sketch; set one before regular long-term use if you want basic protection

VS Code / PlatformIO workflow:
- this folder is now also a PlatformIO project
- `src/main.cpp` simply includes `esp32_grbl_bridge_plotter_hardened.ino`, so Arduino IDE and PlatformIO both use the same source
- OTA target hostname is set to `esp32-grbl-bridge.local`
- fallback OTA target IPs are set to `10.0.0.89` and `10.0.0.90`
- useful tasks in VS Code:
  - `ESP32 OTA Build`
  - `ESP32 OTA Upload (.local)`
  - `ESP32 OTA Upload (10.0.0.89)`
  - `ESP32 OTA Upload (10.0.0.90)`
  - `ESP32 USB Upload`

Terminal equivalents from the repo root:
- `./.venv/bin/platformio run -d firmware/esp32_grbl_bridge_plotter_hardened -e esp32_bridge_ota`
- `./.venv/bin/platformio run -d firmware/esp32_grbl_bridge_plotter_hardened -e esp32_bridge_ota -t upload`
- `./.venv/bin/platformio run -d firmware/esp32_grbl_bridge_plotter_hardened -e esp32_bridge_ota_ip -t upload`
- `./.venv/bin/platformio run -d firmware/esp32_grbl_bridge_plotter_hardened -e esp32_bridge_ota_ip_90 -t upload`

Note:
- this sketch still includes the legacy servo code for compatibility, even though the current machine uses the Z stepper for pen motion
- if we want an even leaner bridge later, the next step would be a true stepper-only bridge variant
