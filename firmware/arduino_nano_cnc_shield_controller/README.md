# Arduino Nano CNC Shield Controller

This firmware is a compact GRBL-like controller for the older Arduino Nano/CNC-shield side of the drawing robot. It drives X, Y, and Z stepper motors directly from STEP/DIR pins and reads one active-low limit switch per axis.

It is meant to sit behind the ESP32 HTTP bridge in this repository. The ESP32 receives HTTP commands from the web app and forwards GRBL-style serial commands to this Nano at `115200` baud.

## Hardware Assumptions

- Arduino Nano / ATmega328P
- CNC shield with A4988 or DRV8825 stepper drivers
- X, Y, and Z stepper motors
- Three normally-open limit switches wired from signal to GND
- Limit inputs use `INPUT_PULLUP`, so a pressed switch reads LOW

Default pins:

| Signal | Pin |
| --- | --- |
| X STEP | D2 |
| Y STEP | D3 |
| Z STEP | D4 |
| X DIR | D5 |
| Y DIR | D6 |
| Z DIR | D7 |
| Driver ENABLE | D8 |
| X limit | D9 |
| Y limit | D10 |
| Z limit | D11 |

Change the `axes` array near the top of `arduino_nano_cnc_shield_controller.ino` if your CNC shield uses a different pinout, steps/mm, axis direction, homing direction, or travel size.

## Supported Commands

- `G0` / `G1` linear moves
- `G20` / `G21` inch/mm mode
- `G90` / `G91` absolute/relative positioning
- `G92` set current work coordinates
- `G4 P...` dwell
- `M3` pen down fallback, mapped to `Z28`
- `M5` pen up fallback, mapped to `Z20`
- `$H`, `$HX`, `$HY`, `$HZ` homing
- `$X` unlock from alarm
- `$I` build info
- `$$` settings report
- `?`, `!`, `~` realtime status/feed-hold/resume
- `$J=...` jogging, including the app's pen jog command

The TypeScript UI normally converts pen up/down to explicit Z moves, so `M3` and `M5` are only a fallback for older G-code.

## Safety

This is intentionally conservative:

- The controller boots in `Alarm` until homed.
- Hard limits are enabled.
- Soft limits are enabled using each axis `maxTravelMm`.
- Feed hold pauses motion but does not cut motor power.

Before running a real drawing:

1. Remove belts or uncouple the machine if possible.
2. Flash the sketch.
3. Confirm each axis steps in the expected direction.
4. Confirm each limit switch changes the status report `Pn:` field.
5. Run `$H` with one hand near machine power.
6. Use small jogs before sending generated art.

For a high-speed router, spindle CNC, or anything with meaningful cutting force, use upstream GRBL instead of this sketch.

## Flashing

Arduino IDE:

1. Open `arduino_nano_cnc_shield_controller.ino`.
2. Select your Nano board and bootloader.
3. Select the serial port.
4. Upload.

PlatformIO, from this folder:

```bash
platformio run -e nanoatmega328new
platformio run -e nanoatmega328new -t upload
```

If upload fails, try the old bootloader environment:

```bash
platformio run -e nanoatmega328
platformio run -e nanoatmega328 -t upload
```
