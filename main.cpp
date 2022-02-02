// AlgorithmCheatSheet.cpp  
#include "Common.h"
#include "AlgorithmCheatSheet.h"
#include "recursion.h"
#include "DP.h"
#include <random> 
#include <chrono>
int main()
{      
    vector<int> nums = { 1,2,3,4,5,6,7,8,9,10 };
    vector<int> shuff = nums;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuff.begin(), shuff.end(),std::default_random_engine(seed));

    BST* tree = new BST(shuff);
    vector<int> ans;
    tree->inorderTraversal(tree->root, ans);
    assert(ans == nums);
} 