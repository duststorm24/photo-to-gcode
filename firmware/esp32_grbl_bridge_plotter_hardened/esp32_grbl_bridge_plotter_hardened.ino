#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include <ESP32Servo.h>
#include <ctype.h>

// Install the "ESP32Servo" library from the Arduino IDE Library Manager.
// This is the hardened GRBL Wi-Fi bridge we want to flash next.
// It keeps compatibility with the existing app while reducing bridge load:
// - smaller rolling log
// - less HTTP command logging
// - slimmer /status payload
// - Wi-Fi sleep disabled for steadier long draws

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif

#ifndef OTA_PASSWORD
#define OTA_PASSWORD ""
#endif

// ---------------------------------------------------------------------------
// Wi-Fi + GRBL bridge settings
// ---------------------------------------------------------------------------

const char *OTA_HOSTNAME = "esp32-grbl-bridge";

constexpr uint32_t USB_DEBUG_BAUD = 115200;
constexpr uint32_t GRBL_BAUD = 115200;
constexpr int GRBL_RX_PIN = 16;  // ESP32 RX2 <- Arduino Uno TX
constexpr int GRBL_TX_PIN = 17;  // ESP32 TX2 -> Arduino Uno RX

// ---------------------------------------------------------------------------
// Servo settings
// ---------------------------------------------------------------------------

constexpr int SERVO_PIN = 18;
constexpr int SERVO_MIN_PULSE_US = 500;
constexpr int SERVO_MAX_PULSE_US = 2500;
constexpr int SERVO_SWEEP_STEP_DEGREES = 1;
constexpr int SERVO_SWEEP_STEP_DELAY_MS = 12;
constexpr int SERVO_SETTLE_DELAY_MS = 120;
constexpr bool SERVO_DETACH_AFTER_MOVE = true;

struct ServoSettings {
  int minAngle = 0;       // Smallest safe mechanical angle.
  int maxAngle = 40;      // Largest safe mechanical angle.
  int upAngle = 0;        // Named "pen up" position.
  int downAngle = 35;     // Named "pen down" position.
  int defaultAngle = 0;   // Startup position after boot.
  int stepAngle = 2;      // Small nudge used during tuning.
  int currentAngle = 0;   // Runtime position.
};

ServoSettings servoSettings;
Servo penServo;
bool servoAttached = false;

// ---------------------------------------------------------------------------
// HTTP server + rolling GRBL log
// ---------------------------------------------------------------------------

WebServer server(80);

constexpr size_t MAX_LOG_LINES = 48;
constexpr bool LOG_HTTP_COMMANDS = false;
constexpr bool LOG_HTTP_REALTIME = false;
String logBuffer[MAX_LOG_LINES];
uint32_t logSeqBuffer[MAX_LOG_LINES];
size_t logCount = 0;
size_t logHead = 0;
uint32_t nextLogSequence = 1;
String currentSerialLine;

String lastCommand = "";
String lastRealtime = "";
String lastStatusLine = "";
String lastServoAction = "startup";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

String jsonEscape(const String &value);

void addCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.sendHeader("Connection", "close");
}

void sendJson(int statusCode, const String &payload) {
  addCorsHeaders();
  server.send(statusCode, "application/json", payload);
}

String buildBridgeHealthJson() {
  String json = "{";
  json += "\"ok\":true";
  json += ",\"firmware\":\"plotter_hardened_v3_0\"";
  json += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += ",\"rssi\":" + String(WiFi.RSSI());
  json += ",\"uptimeMs\":" + String(millis());
  json += ",\"freeHeap\":" + String(ESP.getFreeHeap());
  json += ",\"lastStatusLine\":\"" + jsonEscape(lastStatusLine) + "\"";
  json += "}";
  return json;
}

void handleOptions() {
  addCorsHeaders();
  server.send(204);
}

String jsonEscape(const String &value) {
  String escaped;
  escaped.reserve(value.length() + 8);

  for (size_t i = 0; i < value.length(); ++i) {
    const char c = value[i];
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }

  return escaped;
}

void addLogLine(String line) {
  line.trim();
  if (line.isEmpty()) {
    return;
  }

  Serial.println(line);

  if (line.startsWith("<") && line.endsWith(">")) {
    lastStatusLine = line;
  }

  logBuffer[logHead] = line;
  logSeqBuffer[logHead] = nextLogSequence++;
  logHead = (logHead + 1) % MAX_LOG_LINES;
  if (logCount < MAX_LOG_LINES) {
    ++logCount;
  }
}

void clearLogBuffer() {
  for (size_t i = 0; i < MAX_LOG_LINES; ++i) {
    logBuffer[i] = "";
    logSeqBuffer[i] = 0;
  }
  logCount = 0;
  logHead = 0;
}

String buildLogJsonArray() {
  String json = "[";

  const size_t start = (logCount == MAX_LOG_LINES) ? logHead : 0;
  for (size_t i = 0; i < logCount; ++i) {
    const size_t index = (start + i) % MAX_LOG_LINES;
    if (i > 0) {
      json += ",";
    }
    json += "\"";
    json += jsonEscape(logBuffer[index]);
    json += "\"";
  }

  json += "]";
  return json;
}

String buildLogEntriesJsonArray() {
  String json = "[";

  const size_t start = (logCount == MAX_LOG_LINES) ? logHead : 0;
  for (size_t i = 0; i < logCount; ++i) {
    const size_t index = (start + i) % MAX_LOG_LINES;
    if (i > 0) {
      json += ",";
    }
    json += "{\"seq\":";
    json += String(logSeqBuffer[index]);
    json += ",\"line\":\"";
    json += jsonEscape(logBuffer[index]);
    json += "\"}";
  }

  json += "]";
  return json;
}

String requestBody() {
  if (server.hasArg("plain")) {
    return server.arg("plain");
  }
  return "";
}

bool findJsonValueStart(const String &body, const char *key, int &valueStart) {
  const String token = "\"" + String(key) + "\"";
  const int keyIndex = body.indexOf(token);
  if (keyIndex < 0) {
    return false;
  }

  const int colonIndex = body.indexOf(':', keyIndex + token.length());
  if (colonIndex < 0) {
    return false;
  }

  valueStart = colonIndex + 1;
  while (valueStart < body.length() && isspace(static_cast<unsigned char>(body[valueStart]))) {
    ++valueStart;
  }

  return valueStart < body.length();
}

bool extractJsonStringField(const String &body, const char *key, String &value) {
  int start = 0;
  if (!findJsonValueStart(body, key, start)) {
    return false;
  }

  if (body[start] != '"') {
    return false;
  }

  ++start;
  String parsed;
  bool escaped = false;

  for (int i = start; i < body.length(); ++i) {
    const char c = body[i];

    if (escaped) {
      parsed += c;
      escaped = false;
      continue;
    }

    if (c == '\\') {
      escaped = true;
      continue;
    }

    if (c == '"') {
      value = parsed;
      return true;
    }

    parsed += c;
  }

  return false;
}

bool extractJsonIntField(const String &body, const char *key, int &value) {
  int start = 0;
  if (!findJsonValueStart(body, key, start)) {
    return false;
  }

  bool negative = false;
  if (body[start] == '"') {
    ++start;
  }
  if (body[start] == '-') {
    negative = true;
    ++start;
  }

  int end = start;
  while (end < body.length() && isdigit(static_cast<unsigned char>(body[end]))) {
    ++end;
  }
  if (end == start) {
    return false;
  }

  value = body.substring(start, end).toInt();
  if (negative) {
    value = -value;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Servo helpers
// ---------------------------------------------------------------------------

void normalizeServoSettings() {
  if (servoSettings.minAngle > servoSettings.maxAngle) {
    const int temp = servoSettings.minAngle;
    servoSettings.minAngle = servoSettings.maxAngle;
    servoSettings.maxAngle = temp;
  }

  if (servoSettings.stepAngle < 1) {
    servoSettings.stepAngle = 1;
  }

  servoSettings.defaultAngle = constrain(servoSettings.defaultAngle, servoSettings.minAngle, servoSettings.maxAngle);
  servoSettings.upAngle = constrain(servoSettings.upAngle, servoSettings.minAngle, servoSettings.maxAngle);
  servoSettings.downAngle = constrain(servoSettings.downAngle, servoSettings.minAngle, servoSettings.maxAngle);
  servoSettings.currentAngle = constrain(servoSettings.currentAngle, servoSettings.minAngle, servoSettings.maxAngle);
}

void attachServoIfNeeded() {
  if (servoAttached) {
    return;
  }

  penServo.setPeriodHertz(50);
  penServo.attach(SERVO_PIN, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);
  delay(20);
  servoAttached = true;
}

void detachServoIfNeeded() {
  if (!servoAttached) {
    return;
  }

  penServo.detach();
  servoAttached = false;
}

int clampServoAngle(int angle) {
  return constrain(angle, servoSettings.minAngle, servoSettings.maxAngle);
}

void moveServoToAngle(int targetAngle, const String &reason) {
  normalizeServoSettings();
  attachServoIfNeeded();

  const int clamped = clampServoAngle(targetAngle);
  int current = clampServoAngle(servoSettings.currentAngle);

  if (current != clamped) {
    const int direction = (clamped > current) ? 1 : -1;
    while (current != clamped) {
      current += direction * SERVO_SWEEP_STEP_DEGREES;
      if ((direction > 0 && current > clamped) || (direction < 0 && current < clamped)) {
        current = clamped;
      }
      penServo.write(current);
      delay(SERVO_SWEEP_STEP_DELAY_MS);
      yield();
    }
  } else {
    penServo.write(clamped);
  }

  delay(SERVO_SETTLE_DELAY_MS);
  servoSettings.currentAngle = clamped;
  lastServoAction = reason;

  if (SERVO_DETACH_AFTER_MOVE) {
    detachServoIfNeeded();
  }

  addLogLine("[SERVO] " + reason + " -> angle " + String(clamped) + (SERVO_DETACH_AFTER_MOVE ? " (detached)" : ""));
}

int signedStepForDirection(const String &direction) {
  const bool upMovesTowardLargerAngle = servoSettings.upAngle >= servoSettings.downAngle;
  const int baseStep = max(1, servoSettings.stepAngle);

  if (direction == "up") {
    return upMovesTowardLargerAngle ? baseStep : -baseStep;
  }
  if (direction == "down") {
    return upMovesTowardLargerAngle ? -baseStep : baseStep;
  }
  if (direction == "increase") {
    return baseStep;
  }
  if (direction == "decrease") {
    return -baseStep;
  }

  return 0;
}

bool handlePenMacroCommand(const String &command) {
  String upper = command;
  upper.trim();
  upper.toUpperCase();

  if (upper == "M5" || upper.startsWith("M5 ")) {
    moveServoToAngle(servoSettings.upAngle, "pen up (M5)");
    return true;
  }

  if (upper == "M3" || upper.startsWith("M3 ")) {
    moveServoToAngle(servoSettings.downAngle, "pen down (M3)");
    return true;
  }

  if (upper == "SERVO UP") {
    moveServoToAngle(servoSettings.upAngle, "pen up (manual)");
    return true;
  }

  if (upper == "SERVO DOWN") {
    moveServoToAngle(servoSettings.downAngle, "pen down (manual)");
    return true;
  }

  return false;
}

String buildServoJson() {
  String json = "{";
  json += "\"pin\":" + String(SERVO_PIN);
  json += ",\"minAngle\":" + String(servoSettings.minAngle);
  json += ",\"maxAngle\":" + String(servoSettings.maxAngle);
  json += ",\"upAngle\":" + String(servoSettings.upAngle);
  json += ",\"downAngle\":" + String(servoSettings.downAngle);
  json += ",\"defaultAngle\":" + String(servoSettings.defaultAngle);
  json += ",\"stepAngle\":" + String(servoSettings.stepAngle);
  json += ",\"currentAngle\":" + String(servoSettings.currentAngle);
  json += ",\"attached\":" + String(servoAttached ? "true" : "false");
  json += ",\"lastAction\":\"" + jsonEscape(lastServoAction) + "\"";
  json += "}";
  return json;
}

// ---------------------------------------------------------------------------
// GRBL serial helpers
// ---------------------------------------------------------------------------

void readGrblSerial() {
  while (Serial2.available()) {
    const char c = static_cast<char>(Serial2.read());

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      addLogLine(currentSerialLine);
      currentSerialLine = "";
      continue;
    }

    currentSerialLine += c;
  }
}

void forwardCommandToGrbl(const String &command) {
  lastCommand = command;
  if (LOG_HTTP_COMMANDS) {
    addLogLine("[HTTP->GRBL] " + command);
  }
  Serial2.print(command);
  Serial2.print("\r\n");
}

void forwardRealtimeToGrbl(char realtimeChar) {
  lastRealtime = String(realtimeChar);
  if (LOG_HTTP_REALTIME) {
    addLogLine("[HTTP->GRBL realtime] " + String(realtimeChar));
  }
  Serial2.write(realtimeChar);
}

void warmUpGrbl() {
  Serial.println("Waking up GRBL...");
  Serial2.print("\r\n\r\n");
  delay(500);
  forwardRealtimeToGrbl('?');
  delay(100);
  forwardCommandToGrbl("$I");
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

void handleRoot() {
  addCorsHeaders();

  String html;
  html += "<!doctype html><html><head><meta charset='utf-8'>";
  html += "<title>ESP32 GRBL Bridge</title>";
  html += "<style>body{font-family:sans-serif;max-width:900px;margin:2rem auto;line-height:1.5;padding:0 1rem;}code{background:#f2f2f2;padding:.15rem .35rem;border-radius:4px;}pre{background:#111;color:#eee;padding:1rem;border-radius:8px;overflow:auto;}</style>";
  html += "</head><body>";
  html += "<h1>ESP32 GRBL + Servo Bridge</h1>";
  html += "<p>Bridge IP: <code>" + WiFi.localIP().toString() + "</code></p>";
  html += "<p>Servo pin: <code>GPIO18</code>, safe range <code>" + String(servoSettings.minAngle) + "&deg; - " + String(servoSettings.maxAngle) + "&deg;</code>, current angle <code>" + String(servoSettings.currentAngle) + "&deg;</code>.</p>";
  html += "<h2>HTTP endpoints</h2>";
  html += "<pre>";
  html += "POST /command     { \"cmd\": \"G0 X10 Y10\" }\n";
  html += "POST /realtime    { \"rt\": \"?\" }\n";
  html += "GET  /health\n";
  html += "GET  /status\n";
  html += "POST /restart\n";
  html += "POST /clear-log\n";
  html += "POST /servo/move  { \"name\": \"up\" } or { \"name\": \"down\" }\n";
  html += "POST /servo/nudge { \"direction\": \"up\" } or { \"direction\": \"down\" }\n";
  html += "POST /servo/angle { \"angle\": 87 }\n";
  html += "POST /servo/config { \"minAngle\": 72, \"maxAngle\": 106, \"upAngle\": 78, \"downAngle\": 96, \"defaultAngle\": 78, \"stepAngle\": 2 }\n";
  html += "GET  /servo/status\n";
  html += "</pre>";
  html += "<p>Useful note: this sketch intercepts <code>M3</code> as pen down and <code>M5</code> as pen up, so existing plotter G-code can drive the pen servo through the ESP32.</p>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

void handleCommand() {
  const String body = requestBody();
  String command;

  if (!extractJsonStringField(body, "cmd", command) && server.hasArg("cmd")) {
    command = server.arg("cmd");
  }

  command.trim();
  if (command.isEmpty()) {
    sendJson(400, "{\"ok\":false,\"error\":\"Missing cmd field.\"}");
    return;
  }

  if (handlePenMacroCommand(command)) {
    sendJson(200, "{\"ok\":true,\"intercepted\":true,\"cmd\":\"" + jsonEscape(command) + "\",\"servo\":" + buildServoJson() + "}");
    return;
  }

  forwardCommandToGrbl(command);
  sendJson(200, "{\"ok\":true,\"cmd\":\"" + jsonEscape(command) + "\"}");
}

void handleRealtime() {
  const String body = requestBody();
  String realtime;

  if (!extractJsonStringField(body, "rt", realtime) && server.hasArg("rt")) {
    realtime = server.arg("rt");
  }

  realtime.trim();
  if (realtime.length() != 1) {
    sendJson(400, "{\"ok\":false,\"error\":\"Realtime command must be one character.\"}");
    return;
  }

  const char c = realtime[0];
  if (c != '?' && c != '!' && c != '~') {
    sendJson(400, "{\"ok\":false,\"error\":\"Allowed realtime values are ?, !, and ~.\"}");
    return;
  }

  forwardRealtimeToGrbl(c);
  sendJson(200, "{\"ok\":true,\"rt\":\"" + jsonEscape(realtime) + "\"}");
}

void handleStatus() {
  readGrblSerial();

  String json = "{";
  json += "\"ok\":true";
  json += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += ",\"rssi\":" + String(WiFi.RSSI());
  json += ",\"uptimeMs\":" + String(millis());
  json += ",\"freeHeap\":" + String(ESP.getFreeHeap());
  json += ",\"lastCommand\":\"" + jsonEscape(lastCommand) + "\"";
  json += ",\"lastRealtime\":\"" + jsonEscape(lastRealtime) + "\"";
  json += ",\"lastStatusLine\":\"" + jsonEscape(lastStatusLine) + "\"";
  json += ",\"latestLogSeq\":" + String(nextLogSequence == 0 ? 0 : nextLogSequence - 1);
  json += ",\"recentLogEntries\":" + buildLogEntriesJsonArray();
  json += ",\"recentLog\":" + buildLogJsonArray();
  json += "}";

  sendJson(200, json);
}

void handleHealth() {
  readGrblSerial();
  sendJson(200, buildBridgeHealthJson());
}

void handleRestart() {
  sendJson(200, "{\"ok\":true,\"message\":\"ESP32 bridge restarting\"}");
  delay(150);
  ESP.restart();
}

void handleClearLog() {
  clearLogBuffer();
  sendJson(200, "{\"ok\":true}");
}

void handleServoMove() {
  const String body = requestBody();
  String name;

  if (!extractJsonStringField(body, "name", name)) {
    extractJsonStringField(body, "position", name);
  }

  name.trim();
  name.toLowerCase();

  if (name == "up") {
    moveServoToAngle(servoSettings.upAngle, "named move up");
  } else if (name == "down") {
    moveServoToAngle(servoSettings.downAngle, "named move down");
  } else if (name == "default") {
    moveServoToAngle(servoSettings.defaultAngle, "move to default");
  } else {
    sendJson(400, "{\"ok\":false,\"error\":\"Use name up, down, or default.\"}");
    return;
  }

  sendJson(200, "{\"ok\":true,\"servo\":" + buildServoJson() + "}");
}

void handleServoNudge() {
  const String body = requestBody();
  String direction;
  int steps = 1;

  extractJsonStringField(body, "direction", direction);
  extractJsonIntField(body, "steps", steps);

  direction.trim();
  direction.toLowerCase();
  if (steps < 1) {
    steps = 1;
  }

  const int signedStep = signedStepForDirection(direction);
  if (signedStep == 0) {
    sendJson(400, "{\"ok\":false,\"error\":\"Use direction up, down, increase, or decrease.\"}");
    return;
  }

  moveServoToAngle(
    servoSettings.currentAngle + (signedStep * steps),
    "nudge " + direction + " x" + String(steps)
  );

  sendJson(200, "{\"ok\":true,\"servo\":" + buildServoJson() + "}");
}

void handleServoAngle() {
  const String body = requestBody();
  int angle = 0;

  if (!extractJsonIntField(body, "angle", angle)) {
    sendJson(400, "{\"ok\":false,\"error\":\"Missing integer angle field.\"}");
    return;
  }

  moveServoToAngle(angle, "direct angle test");
  sendJson(200, "{\"ok\":true,\"servo\":" + buildServoJson() + "}");
}

void handleServoConfig() {
  const String body = requestBody();
  int value = 0;

  if (extractJsonIntField(body, "minAngle", value)) {
    servoSettings.minAngle = value;
  }
  if (extractJsonIntField(body, "maxAngle", value)) {
    servoSettings.maxAngle = value;
  }
  if (extractJsonIntField(body, "upAngle", value)) {
    servoSettings.upAngle = value;
  }
  if (extractJsonIntField(body, "downAngle", value)) {
    servoSettings.downAngle = value;
  }
  if (extractJsonIntField(body, "defaultAngle", value)) {
    servoSettings.defaultAngle = value;
  }
  if (extractJsonIntField(body, "stepAngle", value)) {
    servoSettings.stepAngle = value;
  }

  normalizeServoSettings();
  moveServoToAngle(servoSettings.currentAngle, "config clamp");

  sendJson(200, "{\"ok\":true,\"servo\":" + buildServoJson() + "}");
}

void handleServoStatus() {
  sendJson(200, "{\"ok\":true,\"servo\":" + buildServoJson() + "}");
}

void handleNotFound() {
  if (server.method() == HTTP_OPTIONS) {
    handleOptions();
    return;
  }

  sendJson(404, "{\"ok\":false,\"error\":\"Not found.\"}");
}

void registerRoutes() {
  server.on("/", HTTP_GET, handleRoot);

  server.on("/command", HTTP_POST, handleCommand);
  server.on("/command", HTTP_OPTIONS, handleOptions);

  server.on("/realtime", HTTP_POST, handleRealtime);
  server.on("/realtime", HTTP_OPTIONS, handleOptions);

  server.on("/status", HTTP_GET, handleStatus);
  server.on("/status", HTTP_OPTIONS, handleOptions);

  server.on("/health", HTTP_GET, handleHealth);
  server.on("/health", HTTP_OPTIONS, handleOptions);

  server.on("/restart", HTTP_POST, handleRestart);
  server.on("/restart", HTTP_OPTIONS, handleOptions);

  server.on("/clear-log", HTTP_POST, handleClearLog);
  server.on("/clear-log", HTTP_OPTIONS, handleOptions);

  server.on("/servo/move", HTTP_POST, handleServoMove);
  server.on("/servo/move", HTTP_OPTIONS, handleOptions);

  server.on("/servo/nudge", HTTP_POST, handleServoNudge);
  server.on("/servo/nudge", HTTP_OPTIONS, handleOptions);

  server.on("/servo/angle", HTTP_POST, handleServoAngle);
  server.on("/servo/angle", HTTP_OPTIONS, handleOptions);

  server.on("/servo/config", HTTP_POST, handleServoConfig);
  server.on("/servo/config", HTTP_OPTIONS, handleOptions);

  server.on("/servo/status", HTTP_GET, handleServoStatus);
  server.on("/servo/status", HTTP_OPTIONS, handleOptions);

  server.onNotFound(handleNotFound);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(OTA_HOSTNAME);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void setupArduinoOta() {
  ArduinoOTA.setHostname(OTA_HOSTNAME);
  if (strlen(OTA_PASSWORD) > 0) {
    ArduinoOTA.setPassword(OTA_PASSWORD);
  }

  ArduinoOTA.onStart([]() {
    const String mode = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
    addLogLine("[OTA] Start " + mode + " update");
  });

  ArduinoOTA.onEnd([]() {
    addLogLine("[OTA] Update complete");
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    static unsigned int lastPercent = 0;
    if (total == 0) {
      return;
    }
    const unsigned int percent = (progress * 100U) / total;
    if (percent >= lastPercent + 10U || percent == 100U) {
      Serial.printf("[OTA] Progress: %u%%\n", percent);
      lastPercent = percent;
    }
  });

  ArduinoOTA.onError([](ota_error_t error) {
    addLogLine("[OTA] Error " + String(static_cast<int>(error)));
  });

  ArduinoOTA.begin();
  Serial.print("OTA ready on hostname: ");
  Serial.println(OTA_HOSTNAME);
}

void setup() {
  Serial.begin(USB_DEBUG_BAUD);
  delay(200);
  Serial2.begin(GRBL_BAUD, SERIAL_8N1, GRBL_RX_PIN, GRBL_TX_PIN);

  normalizeServoSettings();
  moveServoToAngle(servoSettings.defaultAngle, "startup default");

  connectWiFi();
  setupArduinoOta();
  registerRoutes();
  server.begin();
  Serial.println("HTTP server started.");

  warmUpGrbl();
}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();
  readGrblSerial();
}
