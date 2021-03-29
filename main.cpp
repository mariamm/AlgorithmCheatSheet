// AlgorithmCheatSheet.cpp  
#include "Common.h"
#include "AlgorithmCheatSheet.h"
#include "recursion.h"
#include "DP.h"

int main()
{ 
	string a = "abicideffgi";
	string b = "axbcxdefegx";
	string c = "abcdefg";

	int longest = DP::longestCommonSubsequence(a, b);
	string test = DP::longestCommonSubsequenceReconstructed(a, b);

	cout << c << endl << test << endl<<longest;
	return 0;
} 