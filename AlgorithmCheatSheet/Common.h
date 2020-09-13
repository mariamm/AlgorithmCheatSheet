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
}


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

TreeNode* createTree(const vector<int>& v)
{
    if (v.size() == 0) return NULL;

    TreeNode* root = new TreeNode(v[0]);

    queue<TreeNode*> q;
    q.push(root);

    for (int i = 1; i < v.size(); i+=2)
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