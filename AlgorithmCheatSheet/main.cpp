// AlgorithmCheatSheet.cpp  
#include "Common.h"
#include "AlgorithmCheatSheet.h"
#include "recursion.h"
#include "DP.h"

int main()
{ 
	vector<vector<int>> board = {
			{7,5,5,2,8,0,0,3,0},
			{8,1,0,0,0,9,7,2,5},
			{3,0,2,7,1,5,0,8,9},
			{0,2,1,0,0,8,0,4,0},
			{5,0,8,0,0,0,2,0,0},
			{4,3,0,0,2,0,0,0,8},
			{0,0,0,0,0,0,8,0,0},
			{0,8,0,0,4,1,9,5,2},
			{0,0,0,8,0,2,0,0,0}
		};
	SudokuSolver ss(board);
	vector<vector<int>>solvedboard = ss.solve();
	ss.prettyPrint(board);
	ss.prettyPrint(solvedboard);
} 