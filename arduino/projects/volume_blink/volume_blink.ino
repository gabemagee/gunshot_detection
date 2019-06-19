
/*
 * Blinks when volume goes above a certain threshold 
 */


// SETUP BLOCK----------------------------------

int sensorPin = 0;

void setup() 
{
  pinMode(LED_BUILTIN, OUTPUT);     //Define LED to blink
  pinMode(sensorPin, INPUT);
  Serial.begin(115200);             //Define serial port baud rate
}


// MAIN LOOP------------------------------------
void loop() 
{

long sound[23000];                  // Data array
int shot = 0;                       // Shot positive ID
long i;                             // Counter
i = 0;                              // Initialize counter

  /*
  while(i<23000)
  {
  sound[i++] = analogRead(0);       // Read ADC 23k times into array
  delayMicroseconds(40);            // Wait 40 microseconds between samples
  }
  */

i = 0;                              // Reset counter

int val;
val = analogRead(sensorPin);
Serial.println(val);
if(val > 520 )
{
  digitalWrite(LED_BUILTIN, HIGH);
}
else
{
  digitalWrite(LED_BUILTIN, LOW);
}




/* GEORGE - your code goes here. If your algorithm 
 * detects a gunshot, put the following statement
 * in at the end: 
 * shot = 1; 
 */

 

// GUNSHOT POSITIVE ID LED BLINK-----------------
// Uncomment this to trigger LED on gunshot 
// ----------------------------------------------

/*
if (shot == 1) {                    // If shot detected
  digitalWrite(LED_BUILTIN, HIGH);  // Turn on LED
  delay(500);                       // For 500ms
  digitalWrite(LED_BUILTIN, LOW);   // Turn off LED
  delay(500);                       // Wait 500ms  
}
*/


// SERIAL OUTPUT--------------------------------- 
// Uncomment this to send all data out the serial 
// port for viewing on the Arduino IDE serial plotter
// ----------------------------------------------

/*
i = 0;                              // Reset counter
  while(i<23000)
  {
  Serial.println(sound[i++]);       // Send out all data
  }
  */



}
