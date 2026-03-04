const byte ledPins[] = {2, 3, 4, 5, 6};
const byte NUM_LEDS = 5;

byte lastCount = 255; // sentinel (forces first update)

void setLedCount(byte n) {
  if (n > NUM_LEDS) n = NUM_LEDS;
  for (byte i = 0; i < NUM_LEDS; i++) {
    digitalWrite(ledPins[i], (i < n) ? HIGH : LOW);
  }
}

// Trim leading spaces/tabs
const char* skipSpaces(const char* p) {
  while (*p == ' ' || *p == '\t') p++;
  return p;
}

// Parse "FINGERS:<int>" (with optional whitespace after ':')
bool parseFingerLine(const char* line, int &outVal) {
  const char prefix[] = "FINGERS:";
  const int prefixLen = sizeof(prefix) - 1;

  // Must start with prefix
  for (int i = 0; i < prefixLen; i++) {
    if (line[i] != prefix[i]) return false;
  }

  const char* p = line + prefixLen;
  p = skipSpaces(p);

  // Must have at least one digit
  if (*p < '0' || *p > '9') return false;

  long val = 0;
  while (*p >= '0' && *p <= '9') {
    val = val * 10 + (*p - '0');
    if (val > 1000) return false; // sanity guard
    p++;
  }

  // Allow trailing spaces/tabs only
  p = skipSpaces(p);
  if (*p != '\0') return false;

  outVal = (int)val;
  return true;
}

void setup() {
  Serial.begin(115200);

  for (byte i = 0; i < NUM_LEDS; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }

  Serial.println("Expecting: FINGERS:<count>\\n");
}

void loop() {
  static char buf[48];
  static byte idx = 0;

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\r') continue;        // ignore CR
    if (c == '\n') {                // end of line
      buf[idx] = '\0';
      idx = 0;

      int val;
      if (parseFingerLine(buf, val)) {
        if (val < 0) val = 0;
        if (val > NUM_LEDS) val = NUM_LEDS; // clamp (or reject if you prefer)

        byte count = (byte)val;

        if (count != lastCount) {
          lastCount = count;
          setLedCount(count);

          Serial.print("LEDs on: ");
          Serial.println(count);
        }
      } else {
        // comment out if noisy
        // Serial.print("Bad msg: ");
        // Serial.println(buf);
      }

      continue;
    }

    // accumulate until newline, avoid overflow
    if (idx < sizeof(buf) - 1) buf[idx++] = c;
    else idx = 0; // reset on overflow
  }
}