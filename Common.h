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
#include <sstream>

using namespace std;

//C++ 11 : lambdas, auto type, list initialization, smart pointers, move semantics, range for loops
//C++ 14 : return type deduction (auto functions), binary lieterals (0b), generic lambdas decltypes
//C++ 17 : structured binding (auto [x,y] = a), nested namesapces, inline variables, constexpr lamdas
//C++ 20 : 

///////////////////////////////////////////////
/////// Custom Data Structures ////////////////
///////////////////////////////////////////////
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

struct BST
{
    struct BSTNode
    {
        int val;
        BSTNode* left;
        BSTNode* right;
        int height;

        BSTNode()
        {
            left = nullptr;
            right = nullptr;
            val = 0;
            height = 1;
        }
        BSTNode(int v)
        {
            left = nullptr;
            right = nullptr;
            val = v;
            height = 1;
        }
    };

    BSTNode* root;
    BST()
    {
        root = nullptr;
    }
    BST(int v)
    {
        root = new BSTNode(v);
    }
    BST(vector<int>& values)
    {
        if (values.empty())
        {
            BST();
            return;
        }
        root = new BSTNode(values[0]);
        for (int i = 1; i < values.size(); i++)
        {
            insert(root, values[i]);
        }
    }
    BSTNode* insert(BSTNode* curr, int v)
    {
        if (curr == nullptr) {
            return new BSTNode(v);
        }
        //insert normally
        if (curr->val < v)
        {
            curr->right = insert(curr->right, v);
            curr->right->height++;
        }
        else
        {
            curr->left = insert(curr->left, v);
            curr->left->height++;
        }
        
        //check and fix balance 
        int balance = getBalance(curr);
        /* insert "2"                   
               5            5           0               1   
             3            1              1                3
           2                2             2             2
           LL imbalance  LR imbalance   RR imbalance    RL imbalance
        */
        if (balance > 1)
        { 
            if(curr->left && curr->left->val > v)//LL imbalance
                rotateRight(curr);
            else//LR imbalance
            { 
                rotateRight(curr);
                rotateLeft(curr);
            }
        }
        else if (balance < -1)
        {
            if (curr->right && curr->right->val < v)//RR imbalance
                rotateLeft(curr);
            else//RL imbalance
            {
                rotateLeft(curr);
                rotateRight(curr);
            }

        }
        return curr;
    }
    void rotateLeft(BSTNode* curr)
    {
        BSTNode* temp = curr->right;
        curr->right = nullptr;
        curr->height = 1 + getHeight(curr->left);
        temp->left = curr;
        curr = temp;
        //fix the height
        curr->height = 1 + max(getHeight(curr->left), getHeight(curr->right));

    }
    void rotateRight(BSTNode* curr)
    {
        BSTNode* temp = curr->left;
        curr->left = nullptr;
        curr->height = 1 + getHeight(curr->right);
        temp->right = curr;
        curr = temp;
        //fix the height
        curr->height = 1 + max(getHeight(curr->left), getHeight(curr->right));
    }
    int getBalance(BSTNode *curr)
    {
        return getHeight(curr->left) - getHeight(curr->right);
    }
    int getHeight(BSTNode* curr)
    {
        if (curr == nullptr)
            return 0;
        return curr->height;
    }
    void inorderTraversal(BSTNode* node, vector<int>& arr)
    {
        if (node == nullptr)
            return;
        inorderTraversal(node->left, arr);
        arr.push_back(node->val);
        inorderTraversal(node->right, arr);
    } 
};
struct LinkedList
{
    ListNode* head;  //front
    ListNode* tail;  //back

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
    int frequency;
    Trie()
    {
        frequency = 0;
        isCompleteWord = false;
    }
    void addWord(string word)
    {
        if (word.empty()) return;
        Trie* current = this;
        for (char c : word)
        {
            current->frequency++;
            if (current->children.find(c) == current->children.end())
                current->children[c] = new Trie();

            current = current->children[c];
        }
        current->isCompleteWord = true;
    }
    bool isPrefix(string pref)
    {
        Trie* current = this;
        for (char c : pref)
        {
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return true;
    }
    bool hasWord(string word)
    {
        Trie* current = this;
        for (char c : word)
        {
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return current->isCompleteWord;
    }

    //From leetcode #745
    //prefix sufix combination
    void addWords(const vector<string>& words)
    {
        for (int i = 0; i < words.size(); i++)
        {
            for (int j = words[i].size(); j >= 0; j--)
            {
                //#test", "t#test", "st#test", "est#test", "test#test"
                string subst = words[i].substr(j);
                string temp = subst + "#" + words[i];
                this->addWord(temp);
            }
        }
    }
    //filter by prefix and suffix
    bool findPrefixSuffix(string prefix, string suffix)
    {
        return this->isPrefix(suffix + "#" + prefix);
    }
};

//DSU: Disjoint - Set - Union
//Complexity O(log n) in worst case because of path compression
struct DSU
{
    unordered_map<int, int> parents;
    unordered_map<int, int> size; //union by size path compression
    unordered_map<int, int> rank; //union by rank path compression 

    // 1-based-index
    void make_set(int vertices)
    {
        for (int v = 1; v <= vertices; v++)
        {
            parents[v] = v;
            size[v] = 1;
            rank[v] = 0;
        }
    }
    void make_set(const vector<int>& vertices)
    {
        for (int v : vertices)
        {
            parents[v] = v;
            size[v] = 1;
            rank[v] = 0;
        }
    }
    int find_set(int n)
    {
        if (parents[n] == n)
            return n;
        parents[n] = find_set(parents[n]); //path compression
        return parents[n];
    }
    void union_by_size(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);

        if (a != b) {
            //union by size
            if (size[b] > size[a])
                swap(a, b);

            parents[b] = a;
            size[a] += size[b];
        }
    }
    void union_by_rank(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);

        if (a != b) {
            //union by rank
            if (rank[b] > rank[a])
                swap(a, b);

            parents[b] = a;
            if (rank[a] == rank[b])
                rank[a]++;
        }
    }
};

/* Segment tree
 * Time complexity: construct treeO(nlogn)
 * Time complexity search: O(logn)
 * Space complexity: O(nlogn)
*  Minimum Range
*/
struct SegmentTree
{
    enum class Type {
        Minimum,
        Sum
    };
    SegmentTree(vector<int>& input, Type type)
    {
        m_range = input.size() - 1;
        int n = getSize(input.size());
        if (type == Type::Minimum)
            m_tree.resize(n, INT_MAX);
        else
            m_tree.resize(n);

        m_type = type;
        constructTree(input, 0, 0, m_range);
    }
    void constructTree(vector<int>& input, int pos, int low, int high)
    {
        if (low == high)
            m_tree[pos] = input[low];
        else
        {
            int mid = low + (high - low) / 2;
            constructTree(input, 2 * pos + 1, low, mid);
            constructTree(input, 2 * pos + 2, mid + 1, high);
            m_tree[pos] = operation(m_tree[2 * pos + 1], m_tree[2 * pos + 2]);
        }

    }
    int rangeQuery(int i, int j)
    {
        return rangeQueryHelper(0, m_range, i, j, 0);
    }

    int operation(int i, int j)
    {
        if (m_type == Type::Minimum)
            return min(i, j);
        else if (m_type == Type::Sum)
            return i + j;
        else
            return -1;
    }
private:
    vector<int> m_tree;
    int m_range;
    Type m_type;
    int rangeQueryHelper(int low, int high, int qlow, int qhigh, int pos)
    {
        //full overlap
        if (low >= qlow && high <= qhigh)
            return m_tree[pos];
        //no overlap
        else if (low > qhigh || high < qlow)
            return m_type == Type::Minimum ? INT_MAX : 0;
        //partial overlap
        int mid = low + (high - low) / 2;
        return operation(rangeQueryHelper(low, mid, qlow, qhigh, 2 * pos + 1), rangeQueryHelper(mid + 1, high, qlow, qhigh, 2 * pos + 2));
    }

    int getSize(int inputSize)
    {
        int n = 0;
        for (int i = 0, base = 1; i < 32; i++, base *= 2)
        {
            if (base >= inputSize)
            {
                n = 2 * base - 1;
                break;
            }
        }
        return n;
    }
};
///////////////////////////////////////////////
/////// Data types Constructions //////////////
///////////////////////////////////////////////

ListNode* createLinkedList(const vector<int>& v)
{
    if (v.size() == 0) return NULL;

    ListNode* head = new ListNode(v[0]);
    ListNode* ptr = head;

    for (int i = 1; i < v.size(); i++)
    {
        ptr->next = new ListNode(v[i]);
        ptr = ptr->next;
    }
    return head;
}

vector<int> linkedListToVector(ListNode* head)
{
    vector<int> v;
    while (head)
    {
        v.push_back(head->val);
        head = head->next;
    }
    return v;
}
TreeNode* createTree(const vector<int>& v)
{
    if (v.size() == 0) return NULL;

    TreeNode* root = new TreeNode(v[0]);

    queue<TreeNode*> q;
    q.push(root);

    for (int i = 1; i < v.size(); i += 2)
    {
        TreeNode* p = q.front();
        q.pop();
        if (v[i] != -1)
        {
            p->left = new TreeNode(v[i]);
            q.push(p->left);
        }
        if (v[i + 1] != -1)
        {
            p->right = new TreeNode(v[i]);
            q.push(p->right);
        }
    } 
    return root;
}


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
    cout << "----------------------" << endl;
}

void printVector(vector<int>& input)
{
    for (int i : input)
    {
        cout << i << ", ";
    }
}
