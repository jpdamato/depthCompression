#include "u_ProcessTime.h"
#include <string>
#include <iostream>

#include <time.h>
#include <stdio.h>
#include <mutex>

std::vector<TTimeSignal*> processNames;
bool TimingIsActive = true;

double TTimeSignal::elapsedTime()
{
	double diffticks = _end - _start;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms;
}

void TTimeSignal::startTime()
{
	_start = clock();
}
void TTimeSignal::endTime()
{
	_end = clock();
	double diffticks = _end - _start;
	this->acum += (diffticks * 1000) / CLOCKS_PER_SEC;


}

void stopTimers()
{
	TimingIsActive = false;
}
void startTimers()
{
	TimingIsActive = true;
}

TTimeSignal* root, *prevTS;
std::mutex mtx;

void startProcess(std::string s)
{
	return startProcess((char*)s.c_str());
}


double endProcess(std::string s)
{
	return endProcess((char*)s.c_str());
}

void startProcess(char* procName)
{
	if (!TimingIsActive) return;
	mtx.lock();
	if (root == NULL)
	{
		root = new TTimeSignal((char*)"root");
		prevTS = root;
	}

	TTimeSignal *newT = findProcess(procName);
	if (newT == NULL)
	{
		newT = new TTimeSignal(procName);
		// synchronized creation		
		processNames.push_back(newT);
		
	}
	else
	{
		newT->counter++;
	}
	//Build the invocation tree
	if (std::find(prevTS->childProcess.begin(), prevTS->childProcess.end(), newT) != prevTS->childProcess.end())
	{
	}
	else
	{
		newT->parent = prevTS;
		prevTS->childProcess.push_back(newT);
	}
	prevTS = newT;

	newT->startTime();

	mtx.unlock();
}
double endProcess(char* procName)
{
	
	if (!TimingIsActive) return 0;
	if (processNames.size() == 0) return 0;
	TTimeSignal *newT = findProcess(procName);
	if (newT == NULL)  return 0;
	newT->endTime();
	prevTS = newT->parent;

	return newT->elapsedTime();
}

void showProcessTime()
{
	if (!TimingIsActive) return;
	double totalTime = 0.0;
	std::cout << ".............................................." << "\n";
	std::cout << "...COMPUTED PROCESS TIME......................" << "\n";
	std::cout << ".............................................." << "\n";

	for (unsigned int i = 0; i<processNames.size(); i++)
	{
		TTimeSignal *t = (TTimeSignal*)(processNames[i]);
		//printf("%s%f", t->name, t->elapsedTime());
		if (t->parent == root)
		{
			totalTime += t->acum;
		}
		std::cout << " ..." << t->name << " lastTime " << t->elapsedTime() << " totalTime " << t->acum << "\n";
	}
	std::cout << ".............................................." << "\n";
	std::cout << "          TOTAL TIME : " << totalTime << "\n";
	std::cout << ".............................................." << "\n";
}


void printTreeNode(TTimeSignal* ts, std::string offset, int level, int  depth, double& total)
{

	if (level == 1)
	{
		total += ts->acum;
	}
	if (level == depth) return;
	for (int i = 0; i<ts->childProcess.size(); i++)
	{
		TTimeSignal *t = ts->childProcess[i];
		
		std::cout << offset << t->name << " totalTime " << t->acum << "ms, cnt " << t->counter << "(mean " << t->elapsedTime() << "ms )" << "\n";
		printTreeNode(t, offset + "...", level + 1, depth, total);
	}

}

void showTreeProcessTime(int depth)
{
	if (!TimingIsActive) return;
	double totalTime = 0.0;
	std::cout << ".............................................." << "\n";
	std::cout << "...COMPUTED PROCESS TIME......................" << "\n";
	std::cout << ".............................................." << "\n";
	printTreeNode(root, "", 0, depth, totalTime);
	std::cout << ".............................................." << "\n";
	std::cout << "          TOTAL TIME : " << totalTime << "\n";
	std::cout << ".............................................." << "\n";
}

void clearTimers()
{
	root = NULL;
	for (auto t : processNames)
	{
		free(t);
	}
	processNames.clear();
}


TTimeSignal* findProcess(std::string procName)
{
	for (int i = 0; i<processNames.size(); i++)
	{
		TTimeSignal *t = (TTimeSignal*)(processNames[i]);
		std::string str2(t->name);
		if (procName.compare(str2) == 0)
			return t;
	}
	return NULL;
}

TTimeSignal* findProcess(char* procName)
{
	std::string str1(procName);
	for (int i = 0; i<processNames.size(); i++)
	{
		TTimeSignal *t = (TTimeSignal*)(processNames[i]);
		std::string str2(t->name);
		if (str1.compare(str2) == 0)
			return t;
	}
	return NULL;
}
