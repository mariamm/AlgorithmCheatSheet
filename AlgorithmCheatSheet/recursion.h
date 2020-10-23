#include "Common.h"

/*Recursion (aka Backtracking) Problems*/

/* Basic sudoku solver
 * Assumes input is valid and has a valid solution, 
 * otherwise the same board is the result
 * Empty cells to be filled have value 0
 */
struct SudokuSolver
{
	vector<vector<int>> board;
	vector<unordered_set<int>> rows;  //set for every row in board
	vector<unordered_set<int>> cols;  //set for every column in board
	vector<unordered_set<int>> boxes; //set for every box, linearized
	vector<pair<int, int>> todo;	  //empty cells to be filled
	SudokuSolver(vector<vector<int>>& b) {
		board = b;
		rows.resize(9);
		cols.resize(9);
		boxes.resize(9);

		for (int i = 0; i <9; i++){
			for (int j = 0; j <9; j++){
				if (b[i][j] == 0)
					todo.push_back(make_pair(i, j));
				else{
					rows[i].insert(b[i][j]);
					cols[j].insert(b[i][j]);

					int idx = (i / 3) * 3 + j / 3; //linearized box index
					boxes[idx].insert(b[i][j]);
				}
			}
		}
	}

	bool validEntry(int r, int c, int num)
	{
		if (rows[r].find(num) != rows[r].end())
			return false;

		if (cols[c].find(num) != cols[c].end())
			return false;

		int idx = (r / 3) * 3 + c / 3;

		if (boxes[idx].find(num) != boxes[idx].end())
			return false;

		return true;
	}

	bool recurseSolve(int todoIndex)
	{
		if (todoIndex == todo.size())
			return true;

		int r = todo[todoIndex].first;
		int c = todo[todoIndex].second;
		int idx = (r / 3) * 3 + c / 3;
		for (int i = 1; i <= 9; i++)
		{
			if (validEntry(r, c, i))
			{
				board[r][c] = i;
				rows[r].insert(i);
				cols[c].insert(i);
				boxes[idx].insert(i);
				if (recurseSolve(todoIndex + 1))
					return true;
				else
				{
					board[r][c] = 0;
					rows[r].erase(i);
					cols[c].erase(i);
					boxes[idx].erase(i);
				}
			}
		}
		return false;
	}
	vector<vector<int>> solve()
	{
		recurseSolve(0);
		return board;
	}

	void prettyPrint(vector<vector<int>> board)
	{
		for (int i = 0; i < 9; i++)
		{
			if (i % 3 == 0)
				cout << "---------------------" << endl;
			
			for (int j = 0; j < 9; j++)
			{
				if (j % 3 == 0) 
					cout << "|";
				
				cout << board[i][j] << " ";
			}
			cout << "|" << endl;
		}
		cout << "---------------------" << endl;
	}
};