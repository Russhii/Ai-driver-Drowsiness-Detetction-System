/*
  arduino_pressure_sensor.ino
  ============================
  Steering wheel force sensor — Arduino sketch.

  Hardware:
    - Force Sensitive Resistor (FSR) connected to A0
    - 10kΩ pull-down resistor between A0 and GND
    - NodeMCU ESP8266 or Arduino Uno

  Circuit:
    5V ──── FSR ──── A0 ──── 10kΩ ──── GND

  Serial output:
    Sends raw ADC reading (0–1023) once every 50ms.
    Python reads this and maps it to 0–500 Pa.

  LED indicators (optional):
    Pin 13 (built-in LED): blinks when pressure < threshold
*/

const int FSR_PIN       = A0;
const int LED_PIN       = 13;
const int THRESHOLD_ADC = 409;   // ≈ 200 Pa  (200/500 * 1023)
const int SAMPLE_MS     = 50;    // send reading every 50ms

int   rawValue   = 0;
float smoothed   = 500.0;
bool  gripLost   = false;

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // Read and smooth (low-pass filter)
  rawValue = analogRead(FSR_PIN);
  smoothed = 0.3 * rawValue + 0.7 * smoothed;

  gripLost = (smoothed < THRESHOLD_ADC);

  // LED indicator: blink if grip lost
  if (gripLost) {
    digitalWrite(LED_PIN, (millis() / 300) % 2);   // blink 300ms
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  // Send reading to Python over serial
  Serial.println((int)smoothed);

  delay(SAMPLE_MS);
}

/*
  Wiring Guide
  ─────────────────────────────────────────────────────────
  FSR 402 (round, 0.5") is recommended.
  Place it under the steering wheel grip tape or on
  the 10 o'clock / 2 o'clock hand position.

  To connect to Python:
    1. Plug Arduino via USB
    2. Find port: Windows → Device Manager → COM ports
                  Linux   → ls /dev/ttyUSB*
                  Mac     → ls /dev/cu.usbmodem*
    3. Run:  python main.py --serial-port COM3
       (replace COM3 with your port)

  Threshold calibration:
    - Upload this sketch
    - Open Serial Monitor at 9600 baud
    - Apply normal driving grip → note reading (~400–700)
    - Release grip completely   → note reading (~50–150)
    - Set THRESHOLD_ADC to midpoint
*/
