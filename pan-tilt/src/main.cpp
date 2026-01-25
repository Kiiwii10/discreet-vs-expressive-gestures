// Pan/Tilt SG90 controller with simple 0..180 clamping.
// Commands over Serial:
//   P<deg> T<deg>\n     e.g., "P120 T70"
//   HOME\n              go to home position
//   STATUS\n            print current + target

#include <Arduino.h>

#if defined(ESP32)
  #include <ESP32Servo.h>
  Servo servoPan;
  Servo servoTilt;
#else
  #include <Servo.h>
  Servo servoPan;
  Servo servoTilt;
#endif

#include <Adafruit_NeoPixel.h>

// -------------------- Pins --------------------
static const int PAN_PIN  = 13;
static const int TILT_PIN = 12;

// Addressable LED strips (3 LEDs each).
static const int LED_TOP_PIN = 16;
static const int LED_BOTTOM_PIN = 17;
static const int LED_COUNT = 3;

Adafruit_NeoPixel stripTop(LED_COUNT, LED_TOP_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripBottom(LED_COUNT, LED_BOTTOM_PIN, NEO_GRB + NEO_KHZ800);

static const uint8_t LED_BRIGHTNESS = 80;
static const uint32_t LED_ON_COLOR = 0xFFFFFF;   // white for active motion / clicks
static const uint32_t LED_GUIDE_COLOR = 0x00FF00; // green guidance when idle
static const uint32_t LED_CORRECT_COLOR = 0xFFFF00; // yellow when moving toward target
static const uint32_t LED_REACHED_COLOR = 0x0000FF; // blue when moving within target tolerance

// -------------------- Servo pulse calibration --------------------
// SG90 typical range works with 500..2400us on many boards.
static const int SERVO_MIN_US = 500;
static const int SERVO_MAX_US = 2400;

// -------------------- Home position --------------------
static int PAN_HOME  = 90;
static int TILT_HOME = 60;

// Motion smoothing
static const int STEP_DEG = 1;         // degrees per update step
static const int STEP_DELAY_MS = 8;    // lower = faster, higher = smoother

// State
static int currentPan  = 90;
static int currentTilt = 60;
static int targetPan   = 90;
static int targetTilt  = 60;

static int clampAngle(int v) { return constrain(v, 0, 180); }

static void writeServos(int p, int t) {
  servoPan.write(p);
  servoTilt.write(t);
}

static void goHome() {
  targetPan = clampAngle(PAN_HOME);
  targetTilt = clampAngle(TILT_HOME);
}

static void printStatus() {
  Serial.printf("OK CUR P=%d T=%d | TGT P=%d T=%d | HOME P=%d T=%d\n",
                currentPan, currentTilt,
                targetPan, targetTilt,
                PAN_HOME, TILT_HOME);
}

static uint32_t ledValueToColor(int v) {
  if (v == 2) return LED_GUIDE_COLOR;
  if (v == 1) return LED_ON_COLOR;
  if (v == 3) return LED_CORRECT_COLOR;
  if (v == 4) return LED_REACHED_COLOR;
  return 0;
}

static void setLedGrid(int tl, int tm, int tr, int bl, int bm, int br) {
  stripTop.setPixelColor(0, ledValueToColor(tr));
  stripTop.setPixelColor(1, ledValueToColor(tm));
  stripTop.setPixelColor(2, ledValueToColor(tl));

  stripBottom.setPixelColor(0, ledValueToColor(br));
  stripBottom.setPixelColor(1, ledValueToColor(bm));
  stripBottom.setPixelColor(2, ledValueToColor(bl));

  stripTop.show();
  stripBottom.show();
}

// Parse commands like "P120 T70", "HOME", "STATUS"
static void parseLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  line.toUpperCase();

  if (line == "HOME") {
    goHome();
    Serial.println("OK HOME");
    return;
  }
  if (line == "STATUS") {
    printStatus();
    return;
  }

  if (line.startsWith("LED")) {
    int tl = 0, tm = 0, tr = 0, bl = 0, bm = 0, br = 0;
    if (sscanf(line.c_str(), "LED %d %d %d %d %d %d", &tl, &tm, &tr, &bl, &bm, &br) == 6) {
      setLedGrid(tl, tm, tr, bl, bm, br);
      Serial.println("OK LED");
    } else {
      Serial.println("ERR LED usage: LED <tl> <tm> <tr> <bl> <bm> <br>");
    }
    return;
  }

  // Optional: set home positions:
  //   CENTER <panHome> <tiltHome>
  if (line.startsWith("CENTER")) {
    int firstSpace = line.indexOf(' ');
    if (firstSpace != -1) {
      String rest = line.substring(firstSpace + 1);
      rest.trim();
      int sp = rest.indexOf(' ');
      if (sp != -1) {
        int ph = rest.substring(0, sp).toInt();
        int th = rest.substring(sp + 1).toInt();
        PAN_HOME = clampAngle(ph);
        TILT_HOME = clampAngle(th);
        goHome();
        Serial.printf("OK CENTER panHome=%d tiltHome=%d\n", PAN_HOME, TILT_HOME);
        return;
      }
    }
    Serial.println("ERR CENTER usage: CENTER <panHome> <tiltHome>");
    return;
  }

  // Main command: P<deg> T<deg>
  int pIdx = line.indexOf('P');
  int tIdx = line.indexOf('T');

  bool changed = false;

  if (pIdx != -1) {
    int end = line.indexOf(' ', pIdx);
    if (end == -1) end = line.length();
    int pVal = line.substring(pIdx + 1, end).toInt();
    targetPan = clampAngle(pVal);
    changed = true;
  }

  if (tIdx != -1) {
    int end = line.indexOf(' ', tIdx);
    if (end == -1) end = line.length();
    int tVal = line.substring(tIdx + 1, end).toInt();
    targetTilt = clampAngle(tVal);
    changed = true;
  }

  if (changed) {
    Serial.printf("OK SET P=%d T=%d\n", targetPan, targetTilt);
  } else {
    Serial.println("ERR Unknown command");
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

#if defined(ESP32)
  // ESP32Servo uses LEDC; set 50Hz typical for analog servos.
  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);
  servoPan.attach(PAN_PIN, SERVO_MIN_US, SERVO_MAX_US);
  servoTilt.attach(TILT_PIN, SERVO_MIN_US, SERVO_MAX_US);
#else
  // Servo.h on AVR supports attach(pin, min, max)
  servoPan.attach(PAN_PIN, SERVO_MIN_US, SERVO_MAX_US);
  servoTilt.attach(TILT_PIN, SERVO_MIN_US, SERVO_MAX_US);
#endif

  // Start at home
  currentPan  = clampAngle(PAN_HOME);
  currentTilt = clampAngle(TILT_HOME);
  targetPan   = currentPan;
  targetTilt  = currentTilt;
  writeServos(currentPan, currentTilt);

  stripTop.begin();
  stripBottom.begin();
  stripTop.setBrightness(LED_BRIGHTNESS);
  stripBottom.setBrightness(LED_BRIGHTNESS);
  setLedGrid(0, 0, 0, 0, 0, 0);

  Serial.println("OK READY (send: P<deg> T<deg>, HOME, STATUS, CENTER <p> <t>)");
  printStatus();
}

void loop() {
  // Read a line from Serial (non-blocking-ish)
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseLine(line);
  }

  // Smoothly step toward target
  bool moved = false;

  if (currentPan < targetPan) {
    currentPan += STEP_DEG;
    if (currentPan > targetPan) currentPan = targetPan;
    moved = true;
  } else if (currentPan > targetPan) {
    currentPan -= STEP_DEG;
    if (currentPan < targetPan) currentPan = targetPan;
    moved = true;
  }

  if (currentTilt < targetTilt) {
    currentTilt += STEP_DEG;
    if (currentTilt > targetTilt) currentTilt = targetTilt;
    moved = true;
  } else if (currentTilt > targetTilt) {
    currentTilt -= STEP_DEG;
    if (currentTilt < targetTilt) currentTilt = targetTilt;
    moved = true;
  }

  if (moved) {
    writeServos(currentPan, currentTilt);
    delay(STEP_DELAY_MS);
  }
}
