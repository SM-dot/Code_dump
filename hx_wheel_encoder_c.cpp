#define outputA 6
#define outputB 7

int counter = 0;
int aState;
int aLastState;

void setup() {
    pinMode(outputA, INPUT);
    pinMode(outputB, INPUT);
    Serial.begin(9600);

    // Reads the initial state of outputA
    aLastState = digitalRead(outputA);
}

void loop() {
    aState = digitalRead(outputA);  // Reads the "current" state of outputA

    // If the previous and current state of outputA are different, a Pulse has occurred
    if (aState != aLastState) {
        // If the outputB state is different from the outputA state, the encoder is rotating clockwise
        if (digitalRead(outputB) != aState) {
            counter++;
        } else {
            counter--;
        }

        Serial.print("Position: ");
        Serial.println(counter);
    }

    aLastState = aState;  // Updates the previous state of outputA with the current state
}