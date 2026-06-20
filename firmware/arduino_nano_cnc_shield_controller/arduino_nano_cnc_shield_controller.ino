/*
  Arduino Nano CNC Shield Plotter Controller

  This sketch is a compact GRBL-like controller for a pen plotter using:
  - Arduino Nano / ATmega328P
  - CNC shield with A4988/DRV8825-style STEP/DIR drivers
  - X, Y, and Z stepper axes
  - Three normally-open limit switches wired to ground

  It is designed to work behind the ESP32 HTTP bridge in this repository.
  The supported command subset covers the app's normal draw path:
  G0/G1, G20/G21, G90/G91, G92, G4, M3/M5, $H, $HX/$HY/$HZ, $X, $I,
  $$, and GRBL-style realtime ?, !, ~.

  Safety notes:
  - Keep motors powered off while validating pin directions.
  - Homing is required by default after boot.
  - Limit switches are active-low with INPUT_PULLUP by default.
  - This is not a replacement for full GRBL on a high-speed CNC router.
*/

#include <Arduino.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

constexpr long SERIAL_BAUD = 115200;
constexpr const char *FIRMWARE_VERSION = "nano_cnc_plotter_0.1";

constexpr uint8_t AXIS_COUNT = 3;
constexpr uint8_t AXIS_X = 0;
constexpr uint8_t AXIS_Y = 1;
constexpr uint8_t AXIS_Z = 2;

constexpr uint8_t ENABLE_PIN = 8;
constexpr bool ENABLE_ACTIVE_LOW = true;
constexpr bool LIMIT_SWITCH_ACTIVE_LOW = true;
constexpr bool HARD_LIMITS_ENABLED = true;
constexpr bool HOMING_REQUIRED = true;
constexpr bool SOFT_LIMITS_ENABLED = true;

constexpr unsigned int STEP_PULSE_US = 5;
constexpr unsigned long MIN_STEP_INTERVAL_US = 350;
constexpr float DEFAULT_FEED_MM_MIN = 2400.0f;
constexpr float DEFAULT_RAPID_FEED_MM_MIN = 3600.0f;
constexpr float HOMING_PULL_OFF_MM = 3.0f;
constexpr float PEN_UP_Z_MM = 20.0f;
constexpr float PEN_DOWN_Z_MM = 28.0f;

struct AxisConfig {
  char letter;
  uint8_t stepPin;
  uint8_t dirPin;
  uint8_t limitPin;
  float stepsPerMm;
  float maxTravelMm;
  bool invertDirection;
  int8_t homingDirection;
  float homingFastFeedMmMin;
  float homingSlowFeedMmMin;
};

AxisConfig axes[AXIS_COUNT] = {
  {'X', 2, 5, 9, 80.0f, 300.0f, false, -1, 1800.0f, 420.0f},
  {'Y', 3, 6, 10, 80.0f, 300.0f, false, -1, 1800.0f, 420.0f},
  {'Z', 4, 7, 11, 400.0f, 40.0f, false, -1, 900.0f, 240.0f},
};

enum ControllerState {
  STATE_IDLE,
  STATE_RUN,
  STATE_HOLD,
  STATE_HOME,
  STATE_ALARM,
};

ControllerState controllerState = HOMING_REQUIRED ? STATE_ALARM : STATE_IDLE;

long currentSteps[AXIS_COUNT] = {0, 0, 0};
float currentPositionMm[AXIS_COUNT] = {0.0f, 0.0f, 0.0f};
bool axisHomed[AXIS_COUNT] = {false, false, false};
bool absoluteMode = true;
float unitsScale = 1.0f;
float modalFeedMmMin = DEFAULT_FEED_MM_MIN;
bool feedHoldRequested = false;

constexpr uint8_t MAX_LINE_LENGTH = 110;
char lineBuffer[MAX_LINE_LENGTH + 1];
uint8_t lineLength = 0;

long roundToLong(float value) {
  return static_cast<long>(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

const char *stateName() {
  switch (controllerState) {
    case STATE_IDLE:
      return "Idle";
    case STATE_RUN:
      return "Run";
    case STATE_HOLD:
      return "Hold";
    case STATE_HOME:
      return "Home";
    case STATE_ALARM:
      return "Alarm";
  }
  return "Unknown";
}

void setStepperEnable(bool enabled) {
  digitalWrite(ENABLE_PIN, (enabled == ENABLE_ACTIVE_LOW) ? LOW : HIGH);
}

bool limitPressed(uint8_t axis) {
  const int value = digitalRead(axes[axis].limitPin);
  return LIMIT_SWITCH_ACTIVE_LOW ? value == LOW : value == HIGH;
}

String activeLimitPins() {
  String pins;
  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (limitPressed(axis)) {
      pins += axes[axis].letter;
    }
  }
  return pins;
}

void reportStatus() {
  Serial.print('<');
  Serial.print(stateName());
  Serial.print("|MPos:");
  Serial.print(currentPositionMm[AXIS_X], 3);
  Serial.print(',');
  Serial.print(currentPositionMm[AXIS_Y], 3);
  Serial.print(',');
  Serial.print(currentPositionMm[AXIS_Z], 3);
  Serial.print("|WPos:");
  Serial.print(currentPositionMm[AXIS_X], 3);
  Serial.print(',');
  Serial.print(currentPositionMm[AXIS_Y], 3);
  Serial.print(',');
  Serial.print(currentPositionMm[AXIS_Z], 3);

  const String pins = activeLimitPins();
  if (pins.length() > 0) {
    Serial.print("|Pn:");
    Serial.print(pins);
  }

  Serial.println('>');
}

void printOk() {
  Serial.println(F("ok"));
}

void printError(uint8_t code) {
  Serial.print(F("error:"));
  Serial.println(code);
}

void printAlarm(uint8_t code) {
  controllerState = STATE_ALARM;
  feedHoldRequested = false;
  setStepperEnable(false);
  Serial.print(F("ALARM:"));
  Serial.println(code);
}

void handleRealtime(char c) {
  if (c == '?') {
    reportStatus();
    return;
  }

  if (c == '!') {
    feedHoldRequested = true;
    if (controllerState == STATE_RUN || controllerState == STATE_HOME) {
      controllerState = STATE_HOLD;
    }
    return;
  }

  if (c == '~') {
    feedHoldRequested = false;
    if (controllerState == STATE_HOLD) {
      controllerState = STATE_RUN;
    }
  }
}

void serviceRealtimeOnly() {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.peek());
    if (c != '?' && c != '!' && c != '~') {
      return;
    }
    Serial.read();
    handleRealtime(c);
  }
}

void waitWhileHeld() {
  while (feedHoldRequested) {
    controllerState = STATE_HOLD;
    serviceRealtimeOnly();
    delay(5);
  }
}

void pulseAxes(const bool shouldStep[AXIS_COUNT]) {
  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (shouldStep[axis]) {
      digitalWrite(axes[axis].stepPin, HIGH);
    }
  }
  delayMicroseconds(STEP_PULSE_US);
  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (shouldStep[axis]) {
      digitalWrite(axes[axis].stepPin, LOW);
    }
  }
}

bool hardLimitTripped() {
  if (!HARD_LIMITS_ENABLED || controllerState == STATE_HOME) {
    return false;
  }

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (limitPressed(axis)) {
      return true;
    }
  }
  return false;
}

void setDirection(uint8_t axis, int8_t directionSign) {
  const bool positiveDirection = directionSign >= 0;
  const bool outputHigh = positiveDirection ^ axes[axis].invertDirection;
  digitalWrite(axes[axis].dirPin, outputHigh ? HIGH : LOW);
}

void updatePositionFromSteps() {
  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    currentPositionMm[axis] = static_cast<float>(currentSteps[axis]) / axes[axis].stepsPerMm;
  }
}

bool targetWithinSoftLimits(const float targetMm[AXIS_COUNT]) {
  if (!SOFT_LIMITS_ENABLED) {
    return true;
  }

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (targetMm[axis] < -0.001f || targetMm[axis] > axes[axis].maxTravelMm + 0.001f) {
      return false;
    }
  }
  return true;
}

bool allAxesHomed() {
  return axisHomed[AXIS_X] && axisHomed[AXIS_Y] && axisHomed[AXIS_Z];
}

bool executeLinearMove(const float targetMm[AXIS_COUNT], float feedMmMin, bool rapidMove) {
  if (HOMING_REQUIRED && controllerState == STATE_ALARM) {
    printError(9);
    return false;
  }

  if (!targetWithinSoftLimits(targetMm)) {
    printError(15);
    return false;
  }

  long targetSteps[AXIS_COUNT];
  long deltaSteps[AXIS_COUNT];
  long absSteps[AXIS_COUNT];
  long largestStepCount = 0;
  float deltaMm[AXIS_COUNT];
  float distanceSquared = 0.0f;

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    targetSteps[axis] = roundToLong(targetMm[axis] * axes[axis].stepsPerMm);
    deltaSteps[axis] = targetSteps[axis] - currentSteps[axis];
    absSteps[axis] = labs(deltaSteps[axis]);
    if (absSteps[axis] > largestStepCount) {
      largestStepCount = absSteps[axis];
    }
    deltaMm[axis] = targetMm[axis] - currentPositionMm[axis];
    distanceSquared += deltaMm[axis] * deltaMm[axis];
  }

  if (largestStepCount == 0) {
    for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
      currentPositionMm[axis] = targetMm[axis];
    }
    return true;
  }

  const float distanceMm = sqrt(distanceSquared);
  const float requestedFeed = rapidMove ? DEFAULT_RAPID_FEED_MM_MIN : feedMmMin;
  const float safeFeed = max(1.0f, requestedFeed);
  const float moveSeconds = max(0.001f, (distanceMm / safeFeed) * 60.0f);
  const unsigned long intervalUs = max(
    MIN_STEP_INTERVAL_US,
    static_cast<unsigned long>((moveSeconds * 1000000.0f) / static_cast<float>(largestStepCount))
  );

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    if (deltaSteps[axis] != 0) {
      setDirection(axis, deltaSteps[axis] > 0 ? 1 : -1);
    }
  }
  delayMicroseconds(20);

  setStepperEnable(true);
  controllerState = STATE_RUN;

  long accumulators[AXIS_COUNT] = {0, 0, 0};
  for (long stepIndex = 0; stepIndex < largestStepCount; ++stepIndex) {
    serviceRealtimeOnly();
    waitWhileHeld();
    controllerState = STATE_RUN;

    if (hardLimitTripped()) {
      printAlarm(1);
      return false;
    }

    bool shouldStep[AXIS_COUNT] = {false, false, false};
    for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
      accumulators[axis] += absSteps[axis];
      if (accumulators[axis] >= largestStepCount) {
        accumulators[axis] -= largestStepCount;
        shouldStep[axis] = absSteps[axis] > 0;
        currentSteps[axis] += (deltaSteps[axis] > 0) ? 1 : -1;
      }
    }

    pulseAxes(shouldStep);
    if (intervalUs > STEP_PULSE_US) {
      delayMicroseconds(intervalUs - STEP_PULSE_US);
    }
  }

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    currentSteps[axis] = targetSteps[axis];
    currentPositionMm[axis] = static_cast<float>(targetSteps[axis]) / axes[axis].stepsPerMm;
  }
  controllerState = STATE_IDLE;
  return true;
}

void stepSingleAxis(uint8_t axis, int8_t directionSign, unsigned long intervalUs) {
  setDirection(axis, directionSign);
  delayMicroseconds(20);
  digitalWrite(axes[axis].stepPin, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(axes[axis].stepPin, LOW);
  currentSteps[axis] += directionSign > 0 ? 1 : -1;
  currentPositionMm[axis] = static_cast<float>(currentSteps[axis]) / axes[axis].stepsPerMm;
  if (intervalUs > STEP_PULSE_US) {
    delayMicroseconds(intervalUs - STEP_PULSE_US);
  }
}

unsigned long feedToStepIntervalUs(uint8_t axis, float feedMmMin) {
  const float stepsPerMinute = max(1.0f, feedMmMin * axes[axis].stepsPerMm);
  return max(MIN_STEP_INTERVAL_US, static_cast<unsigned long>(60000000.0f / stepsPerMinute));
}

bool moveAxisFixed(uint8_t axis, int8_t directionSign, float distanceMm, float feedMmMin) {
  const long steps = max(1L, roundToLong(distanceMm * axes[axis].stepsPerMm));
  const unsigned long intervalUs = feedToStepIntervalUs(axis, feedMmMin);

  setStepperEnable(true);
  for (long i = 0; i < steps; ++i) {
    serviceRealtimeOnly();
    waitWhileHeld();
    controllerState = STATE_HOME;
    stepSingleAxis(axis, directionSign, intervalUs);
  }
  return true;
}

bool moveAxisUntilLimit(uint8_t axis, int8_t directionSign, float maxDistanceMm, float feedMmMin) {
  const long maxSteps = max(1L, roundToLong(maxDistanceMm * axes[axis].stepsPerMm));
  const unsigned long intervalUs = feedToStepIntervalUs(axis, feedMmMin);

  setStepperEnable(true);
  for (long i = 0; i < maxSteps; ++i) {
    serviceRealtimeOnly();
    waitWhileHeld();
    controllerState = STATE_HOME;
    if (limitPressed(axis)) {
      return true;
    }
    stepSingleAxis(axis, directionSign, intervalUs);
  }
  return limitPressed(axis);
}

bool homeAxis(uint8_t axis) {
  controllerState = STATE_HOME;
  feedHoldRequested = false;

  const int8_t homeDir = axes[axis].homingDirection < 0 ? -1 : 1;
  const int8_t awayDir = -homeDir;
  const float searchDistanceMm = axes[axis].maxTravelMm + 20.0f;

  if (limitPressed(axis)) {
    moveAxisFixed(axis, awayDir, HOMING_PULL_OFF_MM, axes[axis].homingSlowFeedMmMin);
  }

  if (!moveAxisUntilLimit(axis, homeDir, searchDistanceMm, axes[axis].homingFastFeedMmMin)) {
    printAlarm(8);
    return false;
  }

  moveAxisFixed(axis, awayDir, HOMING_PULL_OFF_MM, axes[axis].homingSlowFeedMmMin);

  if (!moveAxisUntilLimit(axis, homeDir, HOMING_PULL_OFF_MM * 2.5f, axes[axis].homingSlowFeedMmMin)) {
    printAlarm(8);
    return false;
  }

  moveAxisFixed(axis, awayDir, HOMING_PULL_OFF_MM, axes[axis].homingSlowFeedMmMin);

  currentSteps[axis] = roundToLong(HOMING_PULL_OFF_MM * axes[axis].stepsPerMm);
  currentPositionMm[axis] = HOMING_PULL_OFF_MM;
  axisHomed[axis] = true;
  return true;
}

bool homeRequestedAxes(const String &command) {
  bool requested[AXIS_COUNT] = {false, false, false};

  if (command.length() <= 2) {
    requested[AXIS_X] = true;
    requested[AXIS_Y] = true;
    requested[AXIS_Z] = true;
  } else {
    for (uint8_t i = 2; i < command.length(); ++i) {
      const char c = command[i];
      for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
        if (c == axes[axis].letter) {
          requested[axis] = true;
        }
      }
    }
  }

  if (requested[AXIS_Z] && !homeAxis(AXIS_Z)) {
    return false;
  }
  if (requested[AXIS_X] && !homeAxis(AXIS_X)) {
    return false;
  }
  if (requested[AXIS_Y] && !homeAxis(AXIS_Y)) {
    return false;
  }

  controllerState = STATE_IDLE;
  return true;
}

String cleanCommand(const String &raw) {
  String cleaned;
  cleaned.reserve(raw.length());
  bool inParentheses = false;

  for (uint8_t i = 0; i < raw.length(); ++i) {
    char c = raw[i];
    if (c == ';') {
      break;
    }
    if (c == '(') {
      inParentheses = true;
      continue;
    }
    if (c == ')') {
      inParentheses = false;
      continue;
    }
    if (!inParentheses) {
      cleaned += static_cast<char>(toupper(static_cast<unsigned char>(c)));
    }
  }

  cleaned.trim();
  return cleaned;
}

bool parseWordsAndExecute(String command, bool jogCommand) {
  bool hasMotion = false;
  bool hasG92 = false;
  bool hasDwell = false;
  bool hasAxisWord = false;
  bool hasM3 = false;
  bool hasM5 = false;
  bool localAbsoluteMode = jogCommand ? false : absoluteMode;
  bool rapidMove = false;
  float dwellSeconds = 0.0f;
  float targetMm[AXIS_COUNT] = {
    currentPositionMm[AXIS_X],
    currentPositionMm[AXIS_Y],
    currentPositionMm[AXIS_Z],
  };

  if (jogCommand) {
    const int equalsIndex = command.indexOf('=');
    const int commandLength = static_cast<int>(command.length());
    if (equalsIndex < 0 || equalsIndex >= commandLength - 1) {
      printError(3);
      return false;
    }
    command = command.substring(equalsIndex + 1);
  }

  const char *cursor = command.c_str();
  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    if (!isalpha(static_cast<unsigned char>(*cursor))) {
      ++cursor;
      continue;
    }

    const char letter = static_cast<char>(toupper(static_cast<unsigned char>(*cursor)));
    ++cursor;

    char *endPtr = nullptr;
    const float value = static_cast<float>(strtod(cursor, &endPtr));
    if (endPtr == cursor) {
      printError(2);
      return false;
    }
    cursor = endPtr;

    if (letter == 'G') {
      const int code = static_cast<int>(value + 0.5f);
      if (code == 0) {
        hasMotion = true;
        rapidMove = true;
      } else if (code == 1) {
        hasMotion = true;
        rapidMove = false;
      } else if (code == 4) {
        hasDwell = true;
      } else if (code == 20) {
        unitsScale = 25.4f;
      } else if (code == 21) {
        unitsScale = 1.0f;
      } else if (code == 90) {
        localAbsoluteMode = true;
        if (!jogCommand) {
          absoluteMode = true;
        }
      } else if (code == 91) {
        localAbsoluteMode = false;
        if (!jogCommand) {
          absoluteMode = false;
        }
      } else if (code == 92) {
        hasG92 = true;
      }
      continue;
    }

    if (letter == 'M') {
      const int code = static_cast<int>(value + 0.5f);
      hasM3 = code == 3;
      hasM5 = code == 5;
      continue;
    }

    if (letter == 'F') {
      modalFeedMmMin = max(1.0f, value * unitsScale);
      continue;
    }

    if (letter == 'P') {
      dwellSeconds = max(0.0f, value);
      continue;
    }

    for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
      if (letter == axes[axis].letter) {
        const float valueMm = value * unitsScale;
        targetMm[axis] = localAbsoluteMode ? valueMm : currentPositionMm[axis] + valueMm;
        hasAxisWord = true;
      }
    }
  }

  if (hasG92) {
    for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
      currentPositionMm[axis] = targetMm[axis];
      currentSteps[axis] = roundToLong(currentPositionMm[axis] * axes[axis].stepsPerMm);
    }
    return true;
  }

  if (hasM5) {
    targetMm[AXIS_Z] = PEN_UP_Z_MM;
    hasMotion = true;
    hasAxisWord = true;
  } else if (hasM3) {
    targetMm[AXIS_Z] = PEN_DOWN_Z_MM;
    hasMotion = true;
    hasAxisWord = true;
  }

  if (hasDwell) {
    const unsigned long dwellMs = static_cast<unsigned long>(dwellSeconds * 1000.0f);
    const unsigned long startedAt = millis();
    while (millis() - startedAt < dwellMs) {
      serviceRealtimeOnly();
      waitWhileHeld();
      delay(1);
    }
    return true;
  }

  if (jogCommand && hasAxisWord) {
    return executeLinearMove(targetMm, modalFeedMmMin, false);
  }

  if (hasMotion && hasAxisWord) {
    return executeLinearMove(targetMm, modalFeedMmMin, rapidMove);
  }

  return true;
}

void printSettings() {
  Serial.println(F("$0=5 (step pulse, usec)"));
  Serial.println(F("$10=1 (status report mask)"));
  Serial.print(F("$100="));
  Serial.print(axes[AXIS_X].stepsPerMm, 3);
  Serial.println(F(" (x steps/mm)"));
  Serial.print(F("$101="));
  Serial.print(axes[AXIS_Y].stepsPerMm, 3);
  Serial.println(F(" (y steps/mm)"));
  Serial.print(F("$102="));
  Serial.print(axes[AXIS_Z].stepsPerMm, 3);
  Serial.println(F(" (z steps/mm)"));
  Serial.print(F("$130="));
  Serial.print(axes[AXIS_X].maxTravelMm, 3);
  Serial.println(F(" (x max travel, mm)"));
  Serial.print(F("$131="));
  Serial.print(axes[AXIS_Y].maxTravelMm, 3);
  Serial.println(F(" (y max travel, mm)"));
  Serial.print(F("$132="));
  Serial.print(axes[AXIS_Z].maxTravelMm, 3);
  Serial.println(F(" (z max travel, mm)"));
}

void processCommand(const String &rawCommand) {
  const String command = cleanCommand(rawCommand);
  if (command.length() == 0) {
    printOk();
    return;
  }

  if (command == "?") {
    reportStatus();
    return;
  }

  if (command == "$I") {
    Serial.print(F("[VER:"));
    Serial.print(FIRMWARE_VERSION);
    Serial.println(F(":Photo-to-G-code Nano Controller]"));
    Serial.println(F("[OPT:VNMSL,35,254]"));
    printOk();
    return;
  }

  if (command == "$$") {
    printSettings();
    printOk();
    return;
  }

  if (command.startsWith("$H")) {
    if (homeRequestedAxes(command)) {
      printOk();
    } else {
      printError(9);
    }
    return;
  }

  if (command == "$X") {
    feedHoldRequested = false;
    controllerState = STATE_IDLE;
    printOk();
    return;
  }

  if (command.startsWith("$J=")) {
    if (controllerState == STATE_ALARM) {
      printError(9);
      return;
    }
    if (parseWordsAndExecute(command, true)) {
      printOk();
    }
    return;
  }

  if (command[0] == '$') {
    printError(20);
    return;
  }

  if (controllerState == STATE_ALARM) {
    printError(9);
    return;
  }

  if (parseWordsAndExecute(command, false)) {
    printOk();
  }
}

void readSerialCommands() {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());

    if (c == '?' || c == '!' || c == '~') {
      handleRealtime(c);
      continue;
    }

    if (c == '\r' || c == '\n') {
      lineBuffer[lineLength] = '\0';
      const String command(lineBuffer);
      lineLength = 0;
      processCommand(command);
      continue;
    }

    if (lineLength >= MAX_LINE_LENGTH) {
      lineLength = 0;
      printError(14);
      continue;
    }

    lineBuffer[lineLength++] = c;
  }
}

void setupPins() {
  pinMode(ENABLE_PIN, OUTPUT);
  setStepperEnable(false);

  for (uint8_t axis = 0; axis < AXIS_COUNT; ++axis) {
    pinMode(axes[axis].stepPin, OUTPUT);
    pinMode(axes[axis].dirPin, OUTPUT);
    digitalWrite(axes[axis].stepPin, LOW);
    digitalWrite(axes[axis].dirPin, LOW);
    pinMode(axes[axis].limitPin, INPUT_PULLUP);
  }
}

void setup() {
  setupPins();
  Serial.begin(SERIAL_BAUD);
  delay(100);
  Serial.println(F("Grbl 1.1h ['$' for help]"));
  Serial.print(F("[VER:"));
  Serial.print(FIRMWARE_VERSION);
  Serial.println(F(":Photo-to-G-code Nano Controller]"));
  if (HOMING_REQUIRED) {
    Serial.println(F("ALARM:9"));
  }
}

void loop() {
  readSerialCommands();
}
