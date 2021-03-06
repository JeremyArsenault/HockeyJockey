// How to compile for use with python
//$ gcc -shared -Wl,-soname,testlib -o motor_control.so -fPIC motor_control.c -lwiringPi

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>  //Header file for sleep(). man 3 sleep for details. 
#include <pthread.h> //Header for threading
#include <wiringPi.h> //Header for GPIO

// pin labels
const int RIGHT = 11;
const int LEFT = 12;
const int UP_1 = 16;
const int DOWN_1 = 15;
const int UP_2 = 32;
const int DOWN_2 = 31;

// Creates a thread that sleeps for t milliseconds, then stops X
void *stopX(void *t){
    int *time = (int *)t;
    usleep(*time * 1000);
    digitalWrite(LEFT, HIGH);
    digitalWrite(RIGHT, HIGH);
    return NULL; 
}

// Creates a thread that sleeps for t milliseconds, then stops Y
void *stopY(void *t){
    int *time = (int *)t;
    usleep(*time * 1000);
    digitalWrite(UP_1, HIGH);
    digitalWrite(UP_2, HIGH);
    digitalWrite(DOWN_1, HIGH);
    digitalWrite(DOWN_2, HIGH);
    return NULL; 
}

// Setup GPIO pins
void pinSetup(){
    wiringPiSetupPhys();
    pinMode(RIGHT, OUTPUT);
    digitalWrite(RIGHT, HIGH);
    pinMode(LEFT, OUTPUT);
    digitalWrite(LEFT, HIGH);
    pinMode(UP_1, OUTPUT);
    digitalWrite(UP_1, HIGH);
    pinMode(DOWN_1, OUTPUT);
    digitalWrite(DOWN_1, HIGH);
    pinMode(UP_2, OUTPUT);
    digitalWrite(UP_2, HIGH);
    pinMode(DOWN_2, OUTPUT);
    digitalWrite(DOWN_2, HIGH);
}

// Move left for t milliseconds
void moveLeft(int t){
    digitalWrite(LEFT, LOW);
    pthread_t id;
    pthread_create(&id, NULL, stopX, (void*)&t);
}

// Move right for t milliseconds
void moveRight(int t){
    digitalWrite(RIGHT, LOW);
    pthread_t id;
    pthread_create(&id, NULL, stopX, (void*)&t);
}

// Move up for t milliseconds
void moveUp(int t){
    digitalWrite(UP_1, LOW);
    digitalWrite(UP_2, LOW);
    pthread_t id;
    pthread_create(&id, NULL, stopY, (void*)&t);
}

// Move down for t milliseconds
void moveDown(int t){
    digitalWrite(DOWN_1, LOW);
    digitalWrite(DOWN_2, LOW);
    pthread_t id;
    pthread_create(&id, NULL, stopY, (void*)&t);
}

int main(){
    pinSetup();
    return 0;
}


