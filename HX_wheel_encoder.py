from machine import Pin
import time

outputA_pin = 6
outputB_pin = 7

counter = 0
aState = 0
aLastState = 0

outputA = Pin(outputA_pin, Pin.IN)
outputB = Pin(outputB_pin, Pin.IN)

def setup():
    global aLastState
    aLastState = outputA.value()

def loop():
    global counter, aState, aLastState

    aState = outputA.value()

    if aState != aLastState:
        if outputB.value() != aState:
            counter += 1
        else:
            counter -= 1

        print("Position:", counter)

    aLastState = aState

# Setup
setup()

# Main loop
while True:
    loop()
    time.sleep_ms(10)  # Add a small delay to avoid busy-waiting and reduce CPU usage