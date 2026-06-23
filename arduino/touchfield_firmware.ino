/*
 * Project Nexus — TouchField Haptic Device
 * ==========================================
 * Arduino Mega 2560 firmware for a 20×12 solenoid pin-grid (240 pins).
 * Receives JSON position packets over serial from the Python bridge,
 * drives each pin HIGH/LOW to represent live player + ball positions.
 *
 * Author  : Sanchayan (Unwilling-mcu)
 * GitHub  : github.com/Unwilling-mcu/ProjectNexus
 * Board   : Arduino Mega 2560
 * Baud    : 115200
 *
 * Pin Encoding
 * ─────────────
 *   0x00 = empty cell  → solenoid retracted (LOW)
 *   0x01 = home player → solenoid half-rise (PWM 128)
 *   0x02 = away player → solenoid full-rise  (PWM 200)
 *   0x03 = ball        → solenoid vibrating   (PWM toggle 255/0 at 20Hz)
 *   0x04 = referee     → solenoid full-rise + sustained (PWM 255)
 *   0x05 = goal area   → solenoid low pulse   (PWM 60)
 *
 * Hardware Layout
 * ─────────────────
 *   Grid : 20 columns × 12 rows = 240 solenoids
 *   Drive: 3× TLC5940 16-channel LED/solenoid drivers (covers 48 pins each)
 *           → 5× TLC5940 for all 240 (chained SPI)
 *   MCU  : Arduino Mega 2560 (54 digital I/O, SPI on pins 50-53)
 *   Power: 12V 5A switching supply → solenoids via MOSFET array
 *   Audio: DFPlayer Mini module on Serial1 (pins 18/19)
 *   BLE  : HC-05 on Serial2 (pins 16/17) for companion app
 *
 * Packet Format (Serial JSON, newline-delimited)
 * ────────────────────────────────────────────────
 *   {"t":1718000000.123,"grid":[[0,0,1,0,...],[...],...],"event":"goal","score":"1-0"}
 *   grid: 12 rows × 20 cols of uint8 state codes (see Pin Encoding above)
 *   event: optional string — triggers audio narration via DFPlayer
 */

#include <Arduino.h>
#include <ArduinoJson.h>      // v6 — install via Library Manager
#include <SoftwareSerial.h>

// ── Constants ─────────────────────────────────────────────────────
static const uint8_t GRID_COLS   = 20;
static const uint8_t GRID_ROWS   = 12;
static const uint16_t GRID_SIZE  = GRID_COLS * GRID_ROWS;  // 240

// TLC5940 SPI pins (Mega hardware SPI)
static const uint8_t PIN_SPI_CLK  = 52;
static const uint8_t PIN_SPI_MOSI = 51;
static const uint8_t PIN_SPI_CS   = 53;
static const uint8_t PIN_XLAT     = 48;   // TLC5940 latch
static const uint8_t PIN_BLANK    = 49;   // TLC5940 blank (active LOW)
static const uint8_t PIN_GSCLK    = 47;   // grayscale clock

// DFPlayer Mini (audio narration) on Serial1
// Plug: DFPlayer TX → Mega pin 19 (RX1), DFPlayer RX → Mega pin 18 (TX1)
static const uint32_t DFPLAYER_BAUD = 9600;

// Audio track mapping (stored on microSD in DFPlayer)
enum AudioTrack : uint8_t {
  TRACK_GOAL     = 1,
  TRACK_FOUL     = 2,
  TRACK_OFFSIDE  = 3,
  TRACK_HANDBALL = 4,
  TRACK_DIVE     = 5,
  TRACK_CARD_RED = 6,
  TRACK_CARD_YLW = 7,
  TRACK_KICKOFF  = 8,
  TRACK_HALFTIME = 9,
  TRACK_FULLTIME = 10,
};

// PWM duty values for each cell state
static const uint8_t PWM_EMPTY   = 0;
static const uint8_t PWM_HOME    = 128;
static const uint8_t PWM_AWAY    = 200;
static const uint8_t PWM_BALL_HI = 255;
static const uint8_t PWM_BALL_LO = 0;
static const uint8_t PWM_REF     = 255;
static const uint8_t PWM_GOAL    = 60;

// ── State ──────────────────────────────────────────────────────────
static uint8_t  s_grid[GRID_ROWS][GRID_COLS];         // current cell states
static uint8_t  s_pwm[GRID_ROWS][GRID_COLS];          // PWM values sent to TLC
static uint32_t s_ball_toggle_ms  = 0;                // for ball vibration
static bool     s_ball_phase      = false;
static char     s_serial_buf[1024];
static uint16_t s_serial_pos      = 0;
static uint32_t s_last_packet_ms  = 0;
static bool     s_connected       = false;

// ── TLC5940 helpers ────────────────────────────────────────────────
/*
 * TLC5940 drives 16 channels at 12-bit grayscale.
 * We chain 15 ICs (15 × 16 = 240 channels) for all 240 solenoids.
 * For simplicity, we map 8-bit PWM → 12-bit by shifting left 4 bits.
 */
static const uint8_t N_TLC = 15;   // 15 chained TLC5940s

void tlc_begin() {
    pinMode(PIN_SPI_CLK,  OUTPUT);
    pinMode(PIN_SPI_MOSI, OUTPUT);
    pinMode(PIN_SPI_CS,   OUTPUT);
    pinMode(PIN_XLAT,     OUTPUT);
    pinMode(PIN_BLANK,    OUTPUT);
    pinMode(PIN_GSCLK,    OUTPUT);

    digitalWrite(PIN_SPI_CS,  HIGH);
    digitalWrite(PIN_XLAT,    LOW);
    digitalWrite(PIN_BLANK,   HIGH);   // blank = disable during init
    digitalWrite(PIN_GSCLK,   LOW);

    // Init dot correction (all channels max)
    // (abbreviated: in production use full TLC5940 init sequence)
    digitalWrite(PIN_BLANK, LOW);
}

inline void tlc_spi_byte(uint8_t b) {
    for (int8_t i = 7; i >= 0; i--) {
        digitalWrite(PIN_SPI_MOSI, (b >> i) & 1);
        digitalWrite(PIN_SPI_CLK, HIGH);
        digitalWrite(PIN_SPI_CLK, LOW);
    }
}

void tlc_write_all(uint8_t pwm_flat[GRID_SIZE]) {
    /*
     * Push all 240 × 12-bit values to the TLC chain.
     * Each TLC5940 expects 16 channels × 12 bits = 192 bits = 24 bytes,
     * sent MSB-first. We send from last IC to first (daisy-chain order).
     */
    digitalWrite(PIN_SPI_CS, LOW);

    // Pack 8-bit PWM → 12-bit for each of 240 channels
    // Simplified: send high nibble + full byte (12-bit approximation)
    for (int16_t ch = GRID_SIZE - 1; ch >= 0; ch--) {
        uint16_t val12 = (uint16_t)pwm_flat[ch] << 4;
        tlc_spi_byte((val12 >> 8) & 0x0F);
        tlc_spi_byte(val12 & 0xFF);
    }

    digitalWrite(PIN_SPI_CS,  HIGH);
    digitalWrite(PIN_XLAT,    HIGH);
    delayMicroseconds(1);
    digitalWrite(PIN_XLAT,    LOW);
}

// ── DFPlayer Mini ──────────────────────────────────────────────────
void dfplayer_play(uint8_t track) {
    uint8_t cmd[] = {0x7E, 0xFF, 0x06, 0x03, 0x00, 0x00, track, 0x00, 0x00, 0xEF};
    // Checksum bytes (simplified — use full DFPlayer lib in production)
    Serial1.write(cmd, sizeof(cmd));
}

void dfplayer_set_volume(uint8_t vol) {   // 0-30
    uint8_t cmd[] = {0x7E, 0xFF, 0x06, 0x06, 0x00, 0x00, vol, 0x00, 0x00, 0xEF};
    Serial1.write(cmd, sizeof(cmd));
}

// ── Grid logic ────────────────────────────────────────────────────
void grid_clear() {
    memset(s_grid, 0, sizeof(s_grid));
    memset(s_pwm,  0, sizeof(s_pwm));
}

void grid_compute_pwm(uint32_t now_ms) {
    // Update ball vibration phase at ~20Hz
    if (now_ms - s_ball_toggle_ms >= 50) {
        s_ball_phase = !s_ball_phase;
        s_ball_toggle_ms = now_ms;
    }

    for (uint8_t r = 0; r < GRID_ROWS; r++) {
        for (uint8_t c = 0; c < GRID_COLS; c++) {
            switch (s_grid[r][c]) {
                case 0:  s_pwm[r][c] = PWM_EMPTY;  break;
                case 1:  s_pwm[r][c] = PWM_HOME;   break;
                case 2:  s_pwm[r][c] = PWM_AWAY;   break;
                case 3:  s_pwm[r][c] = s_ball_phase ? PWM_BALL_HI : PWM_BALL_LO; break;
                case 4:  s_pwm[r][c] = PWM_REF;    break;
                case 5:  s_pwm[r][c] = PWM_GOAL;   break;
                default: s_pwm[r][c] = PWM_EMPTY;  break;
            }
        }
    }
}

// ── JSON packet parsing ────────────────────────────────────────────
void parse_packet(const char* json_str) {
    StaticJsonDocument<1024> doc;
    DeserializationError err = deserializeJson(doc, json_str);
    if (err) {
        Serial.print(F("[TouchField] JSON error: "));
        Serial.println(err.c_str());
        return;
    }

    // Parse grid
    JsonArray rows = doc["grid"].as<JsonArray>();
    if (!rows.isNull()) {
        uint8_t r = 0;
        for (JsonArray row : rows) {
            uint8_t c = 0;
            for (uint8_t val : row) {
                if (r < GRID_ROWS && c < GRID_COLS)
                    s_grid[r][c] = val;
                c++;
            }
            r++;
            if (r >= GRID_ROWS) break;
        }
    }

    // Play audio narration for game events
    const char* event = doc["event"] | "";
    if      (strcmp(event, "goal")     == 0) dfplayer_play(TRACK_GOAL);
    else if (strcmp(event, "foul")     == 0) dfplayer_play(TRACK_FOUL);
    else if (strcmp(event, "offside")  == 0) dfplayer_play(TRACK_OFFSIDE);
    else if (strcmp(event, "handball") == 0) dfplayer_play(TRACK_HANDBALL);
    else if (strcmp(event, "dive")     == 0) dfplayer_play(TRACK_DIVE);
    else if (strcmp(event, "red_card") == 0) dfplayer_play(TRACK_CARD_RED);
    else if (strcmp(event, "yellow_card") == 0) dfplayer_play(TRACK_CARD_YLW);

    s_last_packet_ms = millis();
    s_connected = true;
}

// ── Serial ingestion ───────────────────────────────────────────────
void serial_ingest() {
    while (Serial.available()) {
        char ch = Serial.read();
        if (ch == '\n') {
            s_serial_buf[s_serial_pos] = '\0';
            if (s_serial_pos > 2) parse_packet(s_serial_buf);
            s_serial_pos = 0;
        } else {
            if (s_serial_pos < sizeof(s_serial_buf) - 1)
                s_serial_buf[s_serial_pos++] = ch;
        }
    }
}

// ── Connection watchdog ────────────────────────────────────────────
void check_connection_timeout() {
    if (s_connected && millis() - s_last_packet_ms > 5000) {
        // No packet for 5s — show "searching" pulse across grid
        s_connected = false;
        for (uint8_t r = 0; r < GRID_ROWS; r++)
            for (uint8_t c = 0; c < GRID_COLS; c++)
                s_grid[r][c] = (r == GRID_ROWS / 2) ? 5 : 0;
    }
}

// ── Arduino entry points ───────────────────────────────────────────
void setup() {
    Serial.begin(115200);     // USB serial — data bridge
    Serial1.begin(DFPLAYER_BAUD);
    Serial2.begin(9600);      // BLE companion app

    tlc_begin();
    grid_clear();

    dfplayer_set_volume(20);
    delay(500);
    dfplayer_play(TRACK_KICKOFF);   // startup chime

    Serial.println(F("[TouchField] Boot complete. Waiting for packets..."));
}

void loop() {
    serial_ingest();
    check_connection_timeout();

    uint32_t now = millis();
    grid_compute_pwm(now);

    // Flatten 2D PWM grid → 1D for TLC chain
    uint8_t flat[GRID_SIZE];
    for (uint8_t r = 0; r < GRID_ROWS; r++)
        for (uint8_t c = 0; c < GRID_COLS; c++)
            flat[r * GRID_COLS + c] = s_pwm[r][c];

    tlc_write_all(flat);

    // ~30fps update rate
    delay(33);
}
