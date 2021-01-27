// AlgorithmCheatSheet.cpp  
#include "Common.h"
#include "AlgorithmCheatSheet.h"
#include "recursion.h"
#include "DP.h"

int main()
{ 
    vector<vector<int>> adjList = { 
        {},
        {2, 3},
        {2, 4, 4},
        {1, 2, 5}, 
        {3, 6},
        {6},
        {3},
    };
    vector<int> solution = hierholzerEulerianPath(adjList, 7);
    printVector(solution);
    int x;
    cin >> x;
} 