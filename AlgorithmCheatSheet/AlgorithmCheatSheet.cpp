// AlgorithmCheatSheet.cpp  

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

struct ListNode
{
    int val;
    ListNode* next;
    ListNode* previous; //for double linked list;
    ListNode()
    {
        val = 0;
        next = NULL;
        previous = NULL;
    }
    ListNode(int v)
    {
        val = v;
        next = NULL;
        previous = NULL;
    }
};

struct TreeNode
{
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode()
    {
        val = 0;
        left = NULL; 
        right = NULL;
    }
    TreeNode(int v)
    {
        val = v;
        left = NULL;
        right = NULL;
    }
};
struct LinkedList
{
    ListNode* head;  //front
    ListNode* tail;	 //back

    ~LinkedList() { /* delete nodes from head to end, NULL ptr*/ }
    LinkedList() { head = NULL; tail = NULL; }
    void clear() {/*clear nodes from head to end, NULL ptr*/ }

    void popBack() 
    {
        if (tail != NULL) {
            if (head == tail)
                head = NULL;

            ListNode* temp = tail;
            tail = tail->previous;

            if (tail != NULL)
                tail->next = NULL;

            delete temp;
            temp = NULL;
        }
    }
    void pushFront(ListNode* newHead) 
    {
        if (head != NULL) {
            newHead->next = head;
            head->previous = newHead;
            head = newHead;
        }
        else {
            head = newHead;
            tail = newHead;
        }
    }
    //emplace_front
    void moveToHead(ListNode* n) 
    {
        if (n != head) {
            n->previous->next = n->next;
            if (n != tail)
                n->next->previous = n->previous;
            else
                tail = n->previous;

            pushFront(n);
        }
    }
};

class Trie {
    unordered_map<char, Trie*> children;
    bool isCompleteWord;
    Trie()
    {
        isCompleteWord = false;
    }
    void addWord(string word) 
    {
        Trie* current = this;
        for (size_t i = 0; i < word.size(); i++)
        {
            char c = word[i];
            if (current->children.find(c) == current->children.end())
            {
                Trie* n = new Trie();
                current->children.insert({ c, n });
            }
            current = current->children[c];
            if (i == word.size() - 1)
                current->isCompleteWord = true;
        }
    }
    bool isPrefix(string pref)
    {
        Trie* current = this;
        for (size_t i = 0; i < pref.size(); i++)
        {
            char c = pref[i];
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return true;
    }
    bool hasWord(string word)
    {
        Trie* current = this;
        for (size_t i = 0; i < word.size(); i++)
        {
            char c = word[i];
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return current->isCompleteWord;
    }
};

//DSU: Disjoint - Set - Union

struct DSU
{
    unordered_map<int, int> parents;
    unordered_map<int, int> size; //union by size
    //unordered_map<int, int> ranks; //union by rank  

    void make_set(const vector<int>& vertices)
    {
        for (int v : vertices)
        {
            parents[v] = v;
            size[v] = 1;
        }
    }
    int find_set(int n)
    {
        if (parents[n] == n)
            return n;
        parents[n] = find_set(parents[n]); //path compression
        return parents[n];
    }
    void dsu_union(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);

        //union by size
        if (size[b] > size[a])
            swap(a, b);

        parents[b] = a;
        size[a] += size[b];
    }
};


////////////////////////////////////////////
////////////   Strings        //////////////
////////////////////////////////////////////


/* KMP Algorithm(find substring index)
 * Knuth Morris Pratt
 * If string B is substring of A, return starting index of substring
*/
int kmpSubstring(string A, string B) 
{
    int prev_start = 0;
    size_t i = prev_start;
    size_t j = 0;
    while (i < A.size() && j < B.size()) 
    {
        if (A[i] == B[j]) 
        {
            i++;
            j++;
        }
        else 
        {
            j = 0;
            i = prev_start;
            prev_start++;
        }
        if (j == B.size())
            return prev_start;
    }
    return -1;
}

//Char conversions
char intToChar(int i)
{
    assert(i >= 0 && i <= 9);
    return i + 'a';
}

int charToInt(char c)
{
    assert(c >= '0' && c <= '9');
    return c - 'a';
}
char uppertoLowerCase(char UC)
{
    assert(UC >= 'A' && UC <= 'Z');
    return UC - 'A' + 'a';
}

char lowertoUpperCase(char lc)
{
    assert(lc >= 'a' && lc <= 'z');
    return lc - 'a' + 'A';
}


////////////////////////////////////////////
////////////   Sorting        //////////////
////////////////////////////////////////////

/*Bucket sort 
* Function to sort arr[] of size n using bucket sort 
* Time Complexity: O(n+k) average case, O(n^2) worst case
* Space Complexity: O(n)
*/
void bucketSort(float arr[], int n) 
{
    /* 1) Create n empty buckets */
    vector<vector<float>> b(n);
    /* 2) Put array elements in different buckets */
    for (int i = 0; i < n; i++) 
    {
        int bi = n * (int)arr[i]; // Index in bucket 
        b[bi].push_back(arr[i]);
    }

    /* 3) Sort individual buckets */
    for (int i = 0; i < n; i++)
        sort(b[i].begin(), b[i].end());

    /* 4) Concatenate all buckets into arr[] */
    int index = 0;
    for (int i = 0; i < n; i++)
        for (size_t j = 0; j < b[i].size(); j++)
            arr[index++] = b[i][j];
}

//Linked List
//Reverse list recursive
ListNode* reverseASingleList(ListNode* A, ListNode*& head)
{
    if (A == NULL)
        return NULL;
    if (A->next == NULL)
    {
        head = A;
        return A;
    }
    ListNode* newNext = reverseASingleList(A->next, head);
    newNext->next = A;
    A->next = NULL;

    return A;
}

//Iterative:
ListNode* reverseList(ListNode* currentNode) 
{ 
    ListNode* previousNode = NULL;
    ListNode* nextNode;

    while (currentNode != NULL) {
        nextNode = currentNode->next;
        currentNode->next = previousNode;
        previousNode = currentNode;
        currentNode = nextNode;
    }
    //loop ends when currentNode == NULL, therefore, reveresed head is previousNode. 
    return previousNode;
}


////////////////////////////////////////////
////////////   Binary search   /////////////
////////////////////////////////////////////

/* Binary search in Array
* Find index of given key in a sorted array. 
* Time Complexity O(log n)
*/
int findKeyInSortedArray(vector<int> &sortedArray, int value)
{
    int low = 0; 
    int high = sortedArray.size() - 1;
    while (low < high)
    {
        int mid = low + (high - low) / 2; //avoid overflow

        if (sortedArray[mid] == value) //found index
            return mid;
    
        if (sortedArray[mid] > value) //search left
            high = mid - 1;
        else //search right
            low = mid + 1;
    }
    return -1;
}

//Binary search sqrt(A)
int sqrt_binarySearch(int A)
{
    if (A == 0) return 0;
    int low = 1;
    int high = A;

    while (low < high)
    {
        long long mid = low + (high - low + 1) / 2;
        if (mid * mid == A)
            return mid;
        if (mid * mid < A)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return low;
}
//Problems
/* Painter Partition problem: minimum time to finish painting.
 * Binary search in rotated array
*/ 

//Binary search tree
//Delete a node from a BST
TreeNode* findMinimum(TreeNode* root) 
{
    TreeNode* current = root;
    while (current->left) {
        current = current->left;
    }
    return current;
}
TreeNode* deleteNode(TreeNode* root, int key) 
{
    if (root == NULL)
        return NULL;

    if (root->val > key)
        root->left = deleteNode(root->left, key);
    else if (root->val < key)
        root->right = deleteNode(root->right, key);

    else //root->val == key 
    {
        if (root->left && root->right) 
        {
            TreeNode* toDelete = findMinimum(root->right); //or find maximum in left subtree 
            root->val = toDelete->val;
            root->right = deleteNode(root->right, root->val);
        }
        else if (root->left || root->right)
        {
            TreeNode* toDelete = root;
            if (root->left)
                root = root->left;
            else
                root = root->right;
            delete toDelete;
        }
        else 
        {
            delete root;
            root = NULL;
        }
    }
    return root;
}
  
//level order traversal for trees 
void levelOrderTraversal(TreeNode* root) 
{
    int current_max_level = 0;
    queue<TreeNode*> q;
    q.push(root);
    int level = 0;
    while (!q.empty()) 
    {
        level++;
        //loop nodes in this level
        for (int i = q.size(); i > 0; --i) 
        {
            TreeNode* n = q.front(); q.pop();
            /*do something*/
            //add children to be processed in next level
            if (n->left)  q.push(n->left);
            if (n->right) q.push(n->right);
        }
    }
}

//Diameter
//Diameter of a tree is the maximum length of a path between two nodes

int treeDepth(TreeNode* root) {
    return (root == NULL) ? 0 : treeDepth(root->left) + treeDepth(root->right) + 1;
}

//Get the maximum path(counting edges) between two nodes(diameter) of a tree, 
//not passing through the root necessarily.

pair<int, int> treeDepthAndDiameter(TreeNode* root) {
    if (root == NULL) return { 0,0 };

    pair<int, int> left = treeDepthAndDiameter(root->left);
    pair<int, int> right = treeDepthAndDiameter(root->right);

    int maxdepth = max(left.first, right.first) + 1;
    int maxdiameter = max({ left.second, right.second, maxdepth });

    return { maxdepth , maxdiameter };
}



////////////////////////////////////////////
////////////   Graphs         //////////////
////////////////////////////////////////////


/* DFS Depth first search graph traversal
*Space Complexity : AdjList O(V + E) AdjMat O(V ^ 2) - 
* Time Complexity: O(V)
*/
//Recursive call. Can be made to return a bool value if we are checking cycles. 
//Iterative implementation would use a stack!
bool dfs_visit(vector<vector<int>>& adj, int s, vector<bool>& parent, vector<bool>& finished)
{
    //startTime[s] =  //TODO
    for (int v : adj[s]) {
        //Node is a child that is not visited = tree edge
        if (parent[v] == false) {
            //Set parent to true, or s if it's a map
            parent[v] = true;
            //Recursive call to visit v
            if (!dfs_visit(adj, v, parent, finished))
                return false; // for cycles
        }

        //Node is visited and finished = forward edge or cross edge  
        //An edge(s,v) is a cross edge, if startTime[s]>startTime[v]. 
       // else if (finished[v])
            /*optional do something*/
        //Node is visited but not finished = backward edge (cycle)
        //else if (!finished[v] && /* parent[v] != vertex for undirected graphs */)
            /*optional do something*/
    }
    finished[s] = true;
    return true;
}
//Main dfs function that calls every vertex in the set (in case of an unconnected vertex) 
void dfs(vector<vector<int>> & adj, vector<int>VerticesSet) {
    //parent aka visited vector. can be a map to know the parent pointers for
    //topological sort!
    vector<bool> parent(VerticesSet.size(), false);
    //optional finished flag vector used to detect cycles
    vector<bool> finished(VerticesSet.size(), false);
    //optional start time vector used to detect cross edges
    vector<int> startTime(VerticesSet.size()); //TODO

    //loop to visit all vertices
    for (int s : VerticesSet) {
        if (!parent[s]) {
            parent[s] = true;
            dfs_visit(adj, s, parent, finished);
        }
    }
}


/*BFS Breadth first search graph traversal 
* Space Complexity : AdjList O(V + E) AdjMat O(V ^ 2) 
* Time Complexity Time : O(V)
* BFS is used for Shortest Path. 
* @param s is starting node. 
* @param t is destination node(optional) 
*/
void BFS(int s, map<int, vector<int>> adj, int t)
{
    queue<int> frontier; //Queue to add children of the visited node
    frontier.push(s);
    unordered_map<int, int> level;//Optional to keep track of level
    level.emplace(s, 0);
    unordered_map<int, int> parent; // aka visited, map is used to extract shorted path
    parent.emplace(s, -1);

    while (!frontier.empty())
    {
        int u = frontier.front();
        frontier.pop(); //remove from queue

        //Print u, do something with u
        /*do something with u*/

        int i = level[u];//level of node
        //Loop adjacent vertices
        for (int v : adj[u])
        {
            //If adjacent nodes are not visited (not in parent)
            if (parent.count(v)) 
            {
                //if (v == t)
                /*destination reached*/ 
                
                parent.insert({ v, u }); //assign u as parent
                level.insert({ v, i + 1 }); //set their level
                frontier.push(v); //add to queue 
            }
        }
    }
}

//Topological sort of a DAG (directed acyclic graph)
vector<int> kahnsort(vector<vector<int>> &graph)
{
    vector<int> indegree(graph.size());
    for (vector<int> n : graph)
    {
        for (int v : n)
        {
            indegree[v]++;
        }
    }
    vector<int> sorted;

    queue<int> q;
    int added = 0;
    for (size_t i = 0; i < indegree.size(); i++)
    {
        if (indegree[i] == 0)
            q.push(i);
        added++;
    }

    while (!q.empty())
    {
        int v = q.front(); q.pop();
        sorted.push_back(v);

        for (int n : graph[v])
        {
            indegree[n]--;
            if (indegree[n] <= 0)
            {
                q.push(n);
                added++;
            }
        }
    }
    /*for(int i : indegree){
        if(i > 0)
            return vector<int>(); //graph has a cycle,
    }
    alternatively  keep an index of added nodes in the q,should match the number of vertices
    if(added != graph.size()) return vector<int>();
    */

    return sorted;
}


////////////////////////////////////////////
////////////   Backtracking   //////////////
////////////////////////////////////////////
 

/*3 Variant Examples for permutation function :
 * Given a collection of numbers, return all possible permutations.
 * @param num : initial vector 
 * @param start : start index for the recursive call
 * @param result : output list of vector permutations
*/
//Variant 1
void permute(vector<int> & num, int start, vector<vector<int> > & result) {
    if (start == num.size() - 1) {
        result.push_back(num);
        return;
    }
    for (size_t i = start; i < num.size(); i++) {
        swap(num[start], num[i]);
        permute(num, start + 1, result);
        swap(num[start], num[i]);
    }
}
/*Variant 2
 *@param num : initial vector
* @param curr : current vector for recursive call
* @param result : output list of vector permutations
*/
void permute(vector<int> & nums, vector<int> & curr, vector<vector<int>> & result) {
    if (curr.size() == nums.size()) {
        result.push_back(curr);
        return;
    }
    for (size_t i = 0; i < nums.size(); i++)
    {
        curr.push_back(nums[i]);
        int temp = nums[i];
        nums[i] = INT_MAX;
        permute(nums, curr, result);
        nums[i] = temp;
        curr.pop_back();
    }
}
//Variant 3 (using std::next_permutation)
void permute(vector<int> &nums, vector<vector<int>> &result)
{
    sort(nums.begin(), nums.end()); //must be sorted in the beginning, otherwise permutations are missing
    do {
        result.push_back(nums);
    } while (next_permutation(nums.begin(), nums.end()));
}
 
////////////////////////////////////////////
////////////   Geometry       //////////////
////////////////////////////////////////////

/*Rotate image by 90 degrees inplace clockwise or counterclockwise
* @param A 2D input image
* @param CCW flag for counterclockwise
*/
void rotateImage(vector<vector<int>> &A, bool CCW = false)
{
    //counter clockwise : swap first, then reverse
    if(!CCW)
        reverse(A.begin(), A.end());
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = i + 1; j < A[0].size(); j++)
            swap(A[i][j], A[j][i]);
    if(CCW)
        reverse(A.begin(), A.end());
}

////////////////////////////////////////////
////////////   Mathematics    //////////////
////////////////////////////////////////////

//Factorial n! = 1*2*3*..*n (optional parameter, start*start+1*...*n) 
unsigned long long int factorial(int n, int start = 1)
{
    unsigned long long int fac = 1;
    for (int i = start + 1; i <= n; i++)
        fac *= i;
    return fac;
}

//Binomial coefficients (N choose k) 
long long nCk(int n, int k)
{
    if (k<0 || k>n) return 0;
    long long numerator = factorial(n, n - k);
    long long denominator = factorial(k);
    return numerator / denominator;
}

int main()
{  
    std::cout << "Hello World!\n";


    int x;
    cin >> x;
} 