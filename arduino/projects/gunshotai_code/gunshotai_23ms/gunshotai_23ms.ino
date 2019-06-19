

// SETUP BLOCK----------------------------------
void setup() 
{
  Serial.begin(115200);             //Define serial port baud rate
  pinMode(LED_BUILTIN, OUTPUT);     //Define LED to blink
}


// MAIN LOOP------------------------------------
void loop() 
{

long sound[23000];                  // Data array
int shot = 0;                       // Shot positive ID
long i;                             // Counter
i = 0;                              // Initialize counter

  
  while(i<23000)
  {
  sound[i++] = analogRead(0);       //Read ADC 23k times into array
  }

i = 0;                              // Reset counter


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
