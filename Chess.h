#pragma once

#include <vector>
#include <memory>
#include <string>
#include <iostream>

using namespace std;
struct ChessBoard;

class ChessPiece
{
	public:
		virtual bool canmove(int row, int col) = 0;
		bool doMove(int row, int col)
		{
			// todo:
			// checkmate
			// en passe
			// promote pawn
			// castle

			bool checkmate = false;
			if (canmove(row, col))
			{
				m_game->board[m_row][m_col] = NULL;
				m_row = row;
				m_col = col;

				delete m_game->board[row][col];
				m_game->board[m_row][m_col] = this;
			}
			return checkmate;
		}
		bool inside(int row, int col)
		{
			return row >= 0 && row < 8 && col >= 0 && col < 8;
		}
		bool black()
		{
			return m_black;
		}
		bool enemy(int row, int col) 
		{
			if (m_game->board[row][col] == NULL)
				return false;

			return m_game->board[row][col]->black() == !m_black;
		}

		bool checkCell(int row, int col)
		{
			//Not in board
			if (!inside(row, col))
				return false;

			//Not a free cell and not enemy cell
			if (m_game->board[row][col] != NULL && !enemy(row, col))
				return false;
		}
	protected:
		shared_ptr<ChessBoard> m_game;
		int m_row, m_col;
		bool m_black; 
		string m_name;
};

class Pawn : public ChessPiece
{
	public:
		Pawn(bool black, int row, int col, shared_ptr<ChessBoard> g)
		{
			m_black = black;
			m_row = row;
			m_col = col;
			m_name = black ? "P_b" : "P_w";
			m_game = g;
		}
		virtual bool canmove(int row, int col) override
		{
			if (!checkCell(row, col))
				return false;

			//regular move forward (down if black, up if white)
			int dir = m_black ? 1 : -1;
			if (col == m_col && row == m_row + dir)
				return true;

			//2 cells allowed in first move
			if (m_black && m_row == 1 || !m_black && m_row == 6)
			{
				if (col == m_col && row == m_row + (2*dir))
					return true;
			}

			if (!enemy(row, col))
				return false;

			//attacking move diagonal
			//left diagonal
			if (col == m_col - 1 && row == m_row + dir)
				return true;

			//right diagonal
			if (col == m_col + 1 && row == m_row + dir)
				return true; 

		}
};

class King : public ChessPiece
{
public:
	King(bool black, int row, int col, shared_ptr<ChessBoard> g)
	{
		m_black = black;
		m_row = row;
		m_col = col;
		m_name = black ? "K_b" : "K_w";
		m_game = g;
	}
};
class Queen : public ChessPiece
{

public:
	Queen(bool black, int row, int col, shared_ptr<ChessBoard> g)
	{
		m_black = black;
		m_row = row;
		m_col = col;
		m_name = black ? "Q_b" : "Q_w";
		m_game = g;
	}
};

class Rook : public ChessPiece
{
public:
	Rook(bool black, int row, int col, shared_ptr<ChessBoard> g)
	{
		m_black = black;
		m_row = row;
		m_col = col;
		m_name = black ? "R_b" : "R_w";
		m_game = g;
	}

};

class Bishop : public ChessPiece
{
public:
	Bishop(bool black, int row, int col, shared_ptr<ChessBoard> g)
	{
		m_black = black;
		m_row = row;
		m_col = col;
		m_name = black ? "B_b" : "B_w";
		m_game = g;
	}

};

class Knight : public ChessPiece
{

public:
	Knight(bool black, int row, int col, shared_ptr<ChessBoard> g)
	{
		m_black = black;
		m_row = row;
		m_col = col;
		m_name = black ? "N_b" : "N_w";
		m_game = g;
	}
};
struct ChessBoard
{
	vector<vector<ChessPiece*>> board;
	bool whiteTurn;

	ChessBoard()
	{
		whiteTurn = true;
		board = vector<vector<ChessPiece*>>(8, vector<ChessPiece*>(8));

		for (int col = 0; col < 8; col++)
		{
			board[1][col] = new Pawn(true, 1, col, make_shared< ChessBoard>(this));
			board[6][col] = new Pawn(false, 6, col, make_shared< ChessBoard>(this));
		}
	}
	void printBoard()
	{

	}
	void play()
	{
		while (true)
		{
			printBoard();
			
			string color = whiteTurn ? "White" : "Black";
			cout <<color<< "'s turn" << endl;
			int row, int col;
			ChessPiece* tomove = nullptr;
			while (tomove == nullptr)
			{
				cin >> row;
				cin >> col;
				tomove = board[row][col];
				if (tomove->black() == whiteTurn)
				{
					tomove = nullptr;
				}
			}
			bool validMove = false;
			while (!validMove)
			{
				cin >> row;
				cin >> col;
				validMove = tomove->canmove(row, col);
			}
			if (tomove->doMove(row, col))
			{
				cout << "Game over. " << color << " wins!";
				return;
			}
			whiteTurn = !whiteTurn;
		}
	}
};