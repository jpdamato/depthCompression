/****************************************************************************

Fit 3D Post-processor
---------------------

u_processTime.h -  library to compute processing time
--------------------------------------------------
Developer : Juan P. D'Amato, PHD.
https://www.researchgate.net/profile/Juan_DAmato/

******************************************************************************/
#include <vector>
#include <time.h>
#include <stdio.h>
#include <vector>
#ifndef TIMESIGNAL
#define TIMESIGNAL

class TTimeSignal
{
public:
	clock_t _start, _end;
	double minTime, maxTime, acum, last, prop;
	std::vector<TTimeSignal*> childProcess;
	int counter;
	TTimeSignal* parent;
	std::string name;
	TTimeSignal(char* name) { this->name= name; acum = 0; counter = 0; }
	void startTime();
	void endTime();
	double elapsedTime(); // in ms

};

void startProcess(char* procName);
void showTreeProcessTime(int depth);
void stopTimers();
void startTimers();
double endProcess(char* procName);


void startProcess(std::string s);
double endProcess(std::string s);

TTimeSignal* findProcess(char* procName);
TTimeSignal* findProcess(std::string procName);

void showProcessTime();
void clearTimers();

#endif