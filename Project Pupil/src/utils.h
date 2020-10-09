#pragma once

#ifndef UTILS_H_
#define UTILS_H_



#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>



using namespace std;


/** Reads string list.
e.g.,
	string dirname = libpath + "data_eye/";
	string filename = "imagelist.txt";
	vector<string> imagelist;
	readStringList(dirname + filename, imagelist);
*/
inline bool readStringList_txt(const string& pathname, vector<string>& stringlist)
{
	stringlist.resize(0);
	ifstream fin(pathname);
	{
		if (!fin.is_open())
			throw("cannot open file");
	}

	string str;
	while (fin >> str)
	{
		stringlist.push_back(str);
	}

	return true;
}


#endif