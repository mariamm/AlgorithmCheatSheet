#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <list>
#include <cassert>
#include <algorithm>

using namespace std;

//C++ 11 : lambdas, auto type, list initialization, smart pointers, move semantics, range for loops
//C++ 14 : return type deduction, binary lieterals, generic lambdas decle types
//C++ 17 : structured binding, nested namesapces, inline variables, constexpr lamdas
//C++ 20 : 

///////////////////////////////////////////////
/////// Pretty Print //////////////////////////
///////////////////////////////////////////////

void print2dVector(vector<vector<int>>& input)
{
    for (vector<int> v : input)
    {
        for (int i : v)
            cout << i << ", ";
        cout << endl;
    }
    cout << "----------------------"<<endl;
}

void printVector(vector<int>& input)
{
    for (int i : input)
    {
        cout << i << ", ";
    }
}