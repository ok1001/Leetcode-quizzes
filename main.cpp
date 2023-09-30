#include <iostream>
#include <string>
#include <cstring>
#include <math.h>
#include <string>
#include <unordered_map>
#include <string_view>
#include <vector>
#include <unordered_set>
#include<algorithm>
#include <stack>
#include <queue>

using namespace std;
#pragma region
class MyStack {
public:
    queue<int>q1, q2;
    MyStack() {

    }

    void push(int x) {
        q2.push(x);
        while (!q1.empty()){
            q2.push(q1.front());
            q1.pop();
        }
        while (!q2.empty()){
            q1.push(q2.front());
            q2.pop();
        }
    }

    int pop() {
        int res = q1.front();
        q1.pop();
        return res;
    }

    int top() {
        return q1.front();
    }

    bool empty() {
        return q1.empty();
    }
};
class MyQueue {
private:
    stack<int> inStack, outStack;
    void in_to_out(){
        while(!inStack.empty()){
            outStack.push(inStack.top());
            inStack.pop();
        }
    };
public:
    MyQueue() {

    }

    void push(int x) {
        inStack.push(x);
    }

    int pop() {
        if(outStack.empty()){
            in_to_out();
        }
        return outStack.top();
    }

    int peek() {
        if(outStack.empty()){
            in_to_out();
        }
        return outStack.top();
    }

    bool empty() {
        return outStack.empty() && inStack.empty();
    }
};
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
class WordDictionary {
    struct Trie{
        unordered_map<char , Trie*>children;
        bool is_word;
        Trie() : is_word(false){}
    };
public:
    Trie *trie = new Trie();
    WordDictionary() {
    }

    void addWord(string word) {
        Trie *cur = trie;
        for(char &c : word){
            if(!cur->children.count(c)){
                cur->children[c] = new Trie();
            }
            cur = cur->children[c];
        }
        cur->is_word = true;
    }

    bool search(string word) {
        return true;
    }
    bool searchHelper(string word, int index, Trie* node){
        if(index == word.size()){
            return node->is_word;
        }
        char c = word[index];
        if(c != '.'){
            if(node->children.count(c)){
                return searchHelper(word, index + 1, node->children[c]);
            } else {
                return false;
            }
        } else{
            for(auto &p : node->children){
                if(searchHelper(word, index + 1, p.second)){
                    return true;
                }
            }
            return false;
        }
    }
};
class MapSum{
private:
    unordered_map<string, int>cnt;
public:
    MapSum(){}
    void insert(string key, int val){
        cnt[key] = val;
    }
    int sum(string prefix){
        int res = 0;
        for (auto & [key, val] : cnt) {
            if(key.substr(0, prefix.size()) == prefix){
                res+= val;
            }
        }
        return res;
    }
};
class Solution1 {
    struct Trie{
        unordered_map<char , Trie*>children;
    };
public:
    string convert(string s, int numRows) {
        if(numRows < 2)
            return s;
        string res;
        int f1 = numRows + numRows - 2;
        int f2 = (s.size() / f1 + 1) * (numRows - 1);
        char a[numRows][f2];
        memset(a, 0, sizeof(a));
        int count = 0;
        for (int j = 0; j < f2 && count < s.size(); j += (numRows - 1)) {
            for (int i = 0; i < numRows && count < s.size(); ++i) {
                a[i][j] = s[count++];
            }
            for (int i = numRows - 2; i > 0 && count < s.size(); --i) {
                int t1 = numRows - i - 1;
                a[i][j + t1] = s[count++];
            }
        }
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < f2; ++j) {
                if(a[i][j] != 0)res += a[i][j];
            }
        }
        return res;
    }

    string replaceWords(vector<string>& dictionary, string sentence) {
        Trie *trie = new Trie();
        for(auto &word : dictionary){
            Trie *cur = trie;
            for(char &c : word){
                if(!cur->children.count(c)){
                    cur->children[c] = new Trie();
                }
                cur = cur->children[c];
            }
            cur->children['#'] = new Trie();
        }
        vector<string> words = split(sentence, ' ');
        for(auto &word : words){
            word = findRoot(word, trie);
        }
        string ans;
        for (int i = 0; i < words.size(); ++i) {
            ans.append(words[i]);
            ans.append(" ");
        }
        ans.append(words.back());
        return ans;
    }
    vector<string> split(string &str, char ch){
        int pos = 0;
        int start = 0;
        vector<string> ret;
        while (pos < str.size()){
            while (pos < str.size() && str[pos] == ch){
                pos++;
            }
            start = pos;
            while (pos < str.size() && str[pos] != ch){
                pos++;
            }
            if (start < str.size()){
                ret.emplace_back(str.substr(start, pos - start));
            }
        }
        return ret;
    }
    string findRoot(string &word, Trie *trie){
        string root;
        Trie *cur = trie;
        for(char &c : word){
            if(cur->children.count('#')){
                return root;
            }
            if(!cur->children.count(c)){
                return word;
            }
            root.push_back(c);
            cur = cur->children[c];
        }
        return root;
    }

    int vowelStrings(vector<string>& words, int left, int right) {
        unordered_set<char > set{'a','e','i','o','u'};
        int res = 0;
        for (int i = left;i < words.size() && i <= right; ++i) {
            char cl = words[i][0];
            char cr = words[i][words[i].size() - 1];
            if(set.count(cl) && set.count(cr))res++;
        }
        return res;
    }
    int maxScore(vector<int>& nums) {
        sort(nums.begin(),nums.end(),greater<int>());
        int res = 0, count = 0;
        for (int i = 0; i < nums.size(); ++i) {
            count += nums[i];
            if(count < 0)return i;
        }
        return nums.size();
    }

    long long beautifulSubarrays(vector<int>& nums) {
        unordered_map<int,int> map;
        int a[nums.size()];
        long long res = 0;
        a[0] = nums[0];
        map[a[0]]++;
        for (int i = 1; i < nums.size(); ++i) {
            a[i] = nums[i] ^ a[i - 1];
            map[a[i]]++;
        }
        for (auto iter = map.begin(); iter != map.end(); ++iter) {
            long long flag = iter->second;
            if(iter->first == 0)res+=iter->second;
            res += flag * (flag - 1) / 2;
        }
        return res;
    }
    void preNode(Node* root, vector<int>& cur){
        if(root == nullptr)return;
        cur.push_back(root->val);
        for (int i = 0; i < root->children.size(); ++i) {
            preNode(root->children[i],cur);
        }
    }
    vector<int> preorder(Node* root) {
        vector<int> res;
        if(root == nullptr){
            return res;
        }
        unordered_map<Node* ,int>cnt;
        stack<Node *>st;
        Node *node = root;
        while(!st.empty() || node != nullptr){
            while (node != nullptr){
                res.emplace_back(node->val);
                st.emplace(node);
                if(node->children.size() > 0){
                    cnt[node] = 0;
                    node = node->children[0];
                } else{
                    node = nullptr;
                }
            }
            node = st.top();
            int index = (cnt.count(node) ? cnt[node] : -1) + 1;
            if(index < node->children.size()){
                cnt[node] = index;
                node = node->children[index];
            }else{
                st.pop();
                cnt.erase(node);
                node = nullptr;
            }
        }
        return res;
    }
    vector<int> postorder(Node* root) {
        vector<int> res;
        if(root == nullptr){
            return res;
        }
        unordered_map<Node* ,int>cnt;
        stack<Node *>st;
        Node *node = root;
        while(!st.empty() || node != nullptr){
            while (node != nullptr){
                st.emplace(node);
                if(node->children.size() > 0){
                    cnt[node] = 0;
                    node = node->children[0];
                } else{
                    node = nullptr;
                }
            }
            node = st.top();
            int index = (cnt.count(node) ? cnt[node] : -1) + 1;
            if(index < node->children.size()){
                cnt[node] = index;
                node = node->children[index];
            }else{
                res.emplace_back(node->val);
                st.pop();
                cnt.erase(node);
                node = nullptr;
            }
        }
        return res;
    }
    vector<vector<int>> levelOrder(Node* root) {
        queue<Node*>obj;
        vector<vector<int>> res;
        if(root == nullptr)return res;
        Node* cur = root;
        obj.push(cur);
        while (obj.size() > 0){
            int n = obj.size();
            vector<int> flag(n);
            for (int i = 0; i < n; ++i) {
                Node* node = obj.front();
                obj.pop();
                flag[i] = node->val;
                for (int j = 0; j < node->children.size(); ++j) {
                    obj.push(node->children[j]);
                }
            }
            res.push_back(flag);
        }
        return res;
    }

    int maxDepth(Node* root) {
        //    int depthmax = 0;
//    void depth(Node* root, int dep){
//        if(dep > depthmax)depthmax = dep;
//        for (int i = 0; i < root->children.size(); ++i) {
//            depth(root->children[i], dep + 1);
//        }
//    }
//    int maxDepth(Node* root) {
//        depth(root, 1);
//        return depthmax;
//    }
        if(root == nullptr){
            return 0;
        }
        int maxChildDepth = 0;
        for (Node* child : root->children) {
            maxChildDepth = max(maxDepth(child), maxChildDepth);
        }
        return maxChildDepth + 1;
    }


    void SumWaysfind(vector<int>& nums, int sum, int target, int count, int &res){
        if(count == nums.size()){
            if(sum == target)res++;
            return;
        }
        SumWaysfind(nums, sum + nums[count],target, count + 1, res);
        SumWaysfind(nums, sum - nums[count],target, count + 1, res);
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        int res = 0;
        if(nums.size() == 1)
            return nums[0] == target;
        SumWaysfind(nums, 0, target, 0, res);
        return res;
    }

    string decode(string s, int & i){
        int flag = s[i] - '0';
        string tmp,res;
        while (i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9'){
            flag = flag * 10 + (s[++i] - '0');
        }
        i += 2;
        while(i < s.size() && s[i] != ']'){
            if(s[i] >= '0' && s[i] <= '9'){
                tmp = tmp + decode(s, i);
            } else{
                tmp = tmp + s[i];
            }
            i++;
        }
        for (int j = 0; j < flag; ++j) {
            res = res + tmp;
        }
        return res;
    }
    string decodeString(string s) {
        string res;
        for (int i = 0; i < s.size(); ++i) {
            if(s[i] >= '0' && s[i] <= '9'){
                res = res + decode(s, i);
            } else{
                res = res + s[i];
            }
        }
        return res;
    }

    vector<vector<int>> floodFillerro(vector<vector<int>>& image, int sr, int sc, int color) {
        queue<vector<int>>q;
        int flag[image.size()][image[0].size()];
        q.push({sr,sc});
        int x = sr, y = sc;
        while (!q.empty()){
            x = q.front()[0];
            y = q.front()[1];
            if(flag[x + 1][y] == 0 && x + 1 < image.size() &&  image[x + 1][y] == image[x][y]){
                q.push({x + 1, y});
                flag[x + 1][y] = 1;
                image[x + 1][y] = color;
            }
            if(flag[x - 1][y] == 0 &&x - 1 < image.size() &&  image[x - 1][y] == image[x][y]){
                q.push({x - 1, y});
                flag[x - 1][y] = 1;
                image[x - 1][y] = color;
            }
            if(flag[x][y + 1] == 0 &&y + 1 < image[0].size() &&  image[x][y + 1] == image[x][y]){
                q.push({x, y + 1});
                flag[x][y + 1] = 1;
                image[x][y + 1] = color;
            }
            if(flag[x][y - 1] == 0 &&y - 1 < image[0].size() &&  image[x][y - 1] == image[x][y]){
                q.push({x, y - 1});
                flag[x][y + 1] = 1;
                image[x][y + 1] = color;
            }
            q.pop();
        }
        return image;
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
        queue<vector<int>> q;
        vector<vector<int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int m = image.size(), n = image[0].size();
        vector<vector<bool>> visited(m,vector<bool>(n, false));
        int oldColor = image[sr][sc];
        q.push({sr, sc});
        visited[sr][sc] = true;
        image[sr][sc] = color;
        while (!q.empty()){
            int x = q.front()[0], y = q.front()[1];
            q.pop();
            for (auto dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && image[nx][ny] == oldColor) {
                    q.push({nx, ny});
                    visited[nx][ny] = true;
                    image[nx][ny] = color;
                }
            }
        }
        return image;
    }
};
struct TreeNode {
         int val;
         TreeNode *left;
         TreeNode *right;
         TreeNode() : val(0), left(nullptr), right(nullptr) {}
         TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
         TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     };
#pragma endregion
class Solution {
#pragma region

private:
    vector<pair<int,int>>freq;
    vector<vector<int>> ans;
    vector<int> sequence;
public:
    static constexpr int directs[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int m = mat.size(), n = mat[0].size();
        queue<pair<int, int>> q;
        vector<vector<int>>visited(m,vector<int>(n));
        vector<vector<int>>res(m,vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if(mat[i][j] == 0){
                    q.emplace(i,j);
                    visited[i][j] = 1;
                }
            }
        }
        while (!q.empty()){
            auto [i, j] = q.front();
            q.pop();
            for(auto dir : directs){
                int nx = i + dir[0], ny = j + dir[1];
                if(nx >= 0 && nx < m && ny >= 0 && ny < n && visited[nx][ny] != 1){
                    res[nx][ny] = res[i][j] + 1;
                    visited[nx][ny] = 1;
                    q.emplace(nx, ny);
                }
            }
        }
        return res;
    }
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        queue<int>q;
        int count = rooms.size();
        vector<bool> visited(rooms.size(),false);
        for (int i = 0; i < rooms[0].size(); ++i) {
            q.push(rooms[0][i]);
        }
        visited[0] = true;
        count--;
        while (!q.empty()){
            int flag = q.front();
            q.pop();
            if(visited[flag] == false){
                for (int i = 0; i < rooms[flag].size(); ++i) {
                    q.push(rooms[flag][i]);
                }
                count--;
                visited[flag] = true;
            }
        }
        if(count == 0)return true;
        return false;
    }

    vector<string> gettarget(string& stat){
        vector<string> ret;
        for (int i = 0; i < 4; ++i) {
            int num = stat[i] - '0';  // 优化1：将字符型数字转为整型数字
            stat[i] = (num + 1) % 10 + '0';  // 优化2：使用数学表达式代替条件判断
            ret.push_back(stat);
            stat[i] = (num + 9) % 10 + '0';
            ret.push_back(stat);
            stat[i] = num + '0';  // 优化3：将字符型数字转回去
        }
        return ret;
    }

    int openLock(vector<string>& deadends, string target) {
        unordered_set<string> dead(deadends.begin(),deadends.end());
        if(target == "0000")return 0;
        if(dead.count("0000"))return -1;
        queue<pair<string,int>> q;
        unordered_set<string> seen = {"0000"};
        q.emplace("0000",0);
        while (!q.empty()){
            auto [status, step] = q.front();
            q.pop();
            for(auto statu : gettarget(status)){
                if(!dead.count(statu) && !seen.count(statu)){
                    if(statu == target)return step + 1;
                    q.push({statu, step + 1});  // 优化4：使用花括号代替圆括号
                    seen.emplace(statu);
                }
            }
        }
        return -1;
    }

    int findMaximumXOR(vector<int>& nums) {
        int x= 0;
        for (int k = 30; k > 0 ; --k) {
            unordered_set<int> seen;
            for(int num : nums){
                seen.insert(num >> k);
            }

            int x_next = x * 2 + 1;
            bool found = false;
            for(int num : nums){
                if (seen.count(x_next ^ (num >> k))){
                    found = true;
                    break;
                }
            }
            if(found){
                x = x_next;
            } else{
                x = x_next - 1;
            }
        }
        return x;
    }

    int majorityElement(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        return nums[nums.size()/2];
    }

    void combindfs(vector<int>& candidates, int begin, vector<vector<int>>& paths,vector<int>& path, int target){
        if(target < 0)return;
        if(target == 0){
            paths.emplace_back(path);
        }
        for (int i = begin; i < candidates.size(); ++i) {
            if(target - candidates[i] < 0)break;
            path.emplace_back(candidates[i]);
            combindfs(candidates,i,paths,path,target -  candidates[i]);
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<vector<int>> res;
        vector<int> path;
        combindfs(candidates, 0, res, path, target);
        return res;
    }
    void dfs(int pos, int rest){
        if(rest == 0){
            ans.push_back(sequence);
            return;
        }
        if(pos == freq.size() || rest < freq[pos].first){
            return;
        }
        dfs(pos + 1, rest);
        int most = min(rest / freq[pos].first, freq[pos].second);
        for (int i = 1; i <= most; ++i) {
            sequence.push_back(freq[pos].first);
            dfs(pos + 1, rest - i * freq[pos].first);
        }
        for (int i = 1; i <= most; ++i) {
            sequence.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        for(int num : candidates){
            if(freq.empty() || num != freq.back().first){
                freq.emplace_back(num, 1);
            } else{
                ++freq.back().second;
            }
        }
        dfs(0,target);
        return ans;
    }

    void backtrack(vector<vector<int>>& res, vector<int>& output, int first, int len){
        if(first == len){
            res.emplace_back(output);
            return;
        }
        for (int i = first; i < len; ++i) {
            swap(output[i], output[first]);
            backtrack(res, output, first + 1, len);
            swap(output[i],output[first]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>>res;
        backtrack(res, nums, 0, (int)nums.size());
        return res;
    }
    vector<vector<int>> pathret;
    vector<int> pathpath;
    void pathsumdfs(TreeNode* root, int targetSum){
        if (root == nullptr){
            return;
        }
        pathpath.emplace_back(root->val);
        targetSum -= root->val;
        if(root->right == nullptr && root->left == nullptr && targetSum == 0){
            pathret.emplace_back(pathpath);
        }
        pathsumdfs(root->right,targetSum);
        pathsumdfs(root->left,targetSum);
        pathpath.pop_back();
    }
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        pathsumdfs(root, targetSum);
        return pathret;
    }

    int maxRepOpt1(string text) {
        unordered_map<char,int> count;
        for(auto c : text){
            count[c]++;
        }
        int res = 0;
        for (int i = 0; i < text.size(); ) {
            int j = i;
            while (j < text.size() && text[j] == text[i]){
                j++;
            }
            int cur_cnt = j - i;
            if(cur_cnt < count[text[i]] && (j < text.size() || i > 0)){
                res = max(res, cur_cnt + 1);
            }
            int k = j + 1;
            while (k < text.size() && text[k] == text[i]){
                k++;
            }
            res = max(res, min(k - i,count[text[i]]));
            i = j;
        }
        return res;
    }
    class Automaton{
        string state = "start";
        unordered_map<string, vector<string>> table = {
                {"start",{"start", "signed", "in_number", "end"}},
                {"signed",{"end", "end", "in_number", "end"}},
                {"in_number",{"end", "end", "in_number", "end"}},
                {"end",{"end", "end", "end", "end"}},
        };
         int get_col( char c){
             if(isspace(c))return 0;
             if(c == '+' or c == '-')return 1;
             if(isdigit(c))return 2;
             return 3;
         }

    public:
        int sign = 1;
         long long ans = 0;
         void get(char c){
             state = table[state][get_col(c)];
             if (state == "in_number"){
                 ans = ans * 10 + c - '0';
                 ans = sign == 1 ? min(ans,(long long)INT_MAX) : min(ans, -(long long)INT_MIN);
             } else if(state == "signed")
                 sign = c == '+' ? 1 : -1;
         }
    };
    public:
        int myAtoi(string str){
            Automaton automaton;
            for (char c : str) {
                automaton.get(c);
            }
            return automaton.sign * automaton.ans;
        }

        vector<int> temp;
        vector<vector<int>> dfsans;
        void dfs(int cur, int n, int k){
            if(temp.size() + (n - cur + 1) < k){
                return;
            }
            if(temp.size() == k){
                dfsans.push_back(temp);
                return;
            }
            temp.push_back(cur);
            dfs(cur + 1, n, k);
            temp.pop_back();
            dfs(cur + 1, n, k);
        }
        vector<vector<int>> combine(int n, int k){
            dfs(1, n, k);
            return dfsans;
        }

        bool isAdditiveNumber(string num) {
            int n = num.size();
            for(int secondStart = 1; secondStart < n -1; ++secondStart){
                if(num[0] == '0' && secondStart != 1){
                    break;
                }
                for(int secondEnd = secondStart;secondEnd < n - 1; ++secondEnd){
                    if(num[secondStart] == '0' && secondStart != secondEnd){
                        break;
                    }
                    if(valid(secondStart, secondEnd, num)){
                        return true;
                    }
                }
            }
            return false;
        }
        bool valid(int secondStart, int secondEnd, string num) {
            int n = num.size();
            int firstStart = 0, firstEnd = secondStart - 1;
            while (secondEnd <= n - 1){
                string third = stringAdd(num, firstStart, firstEnd, secondStart, secondEnd);
                int thirdStart = secondEnd + 1;
                int thirdEnd = secondEnd + third.size();
                if(thirdEnd >= n || !(num.substr(thirdStart, thirdEnd - thirdStart + 1) == third)){
                    break;
                }
                if(thirdEnd == n -1){
                    return true;
                }
                firstStart = secondStart;
                firstEnd = secondEnd;
                secondStart = thirdStart;
                secondEnd = thirdEnd;
          }
            return false;
        };
        string stringAdd(string s, int firstStart, int firstEnd, int secondStart, int secondEnd) {
            string third;
            int carry = 0, cur = 0;
            while (firstEnd >= firstStart || secondEnd >= secondStart || carry != 0){
                cur = carry;
                if(firstEnd >= firstStart){
                    cur += s[firstEnd] - '0';
                    --firstEnd;
                }
                if(secondEnd >= secondStart){
                    cur += s[secondEnd] - '0';
                    --secondEnd;
                }
                carry = cur / 10;
                cur %= 10;
                third.push_back(cur + '0');
            }
            reverse(third.begin(), third.end());
            return third;
        }

        vector<int> t;
        vector<vector<int>> ans1;
        vector<vector<int>> subsets2(vector<int>& nums) {
            int n = nums.size();
            for(int mask = 0; mask < (1 << n); ++mask){
                t.clear();
                for (int i = 0; i < n; ++i) {
                    if(mask & (1 << i)){
                        t.push_back(nums[i]);
                    }
                }
                ans1.push_back(t);
            }
            return ans1;
        }
        void ubsetsdfs(int cur, vector<int>& nums){
            if(cur == nums.size()){
                ans.push_back(t);
                return;
            }
            t.push_back(nums[cur]);
            ubsetsdfs(cur + 1, nums);
            t.pop_back();
            ubsetsdfs(cur + 1, nums);
        }
        vector<vector<int>> subsets3(vector<int>& nums) {
            ubsetsdfs(0,nums);
            return ans1;
        }
         void plusOnedfs(vector<int>& digits, int n) {
        digits[n] = 0;
        if (n == 0)digits.insert(digits.begin(),1);
        if (digits[n - 1] == 9){
            plusOnedfs(digits, n -1);
        } else{
            digits[n]++;
        }
    }
         vector<int> plusOne(vector<int>& digits) {
        int n = digits.size();
        if(digits[n - 1] == 9){
            plusOnedfs(digits, n - 1);
        } else{
            digits[n - 1] ++;
        }
        return digits;
    }
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> res;
        int left, right = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (i < nums.size() - 1 && nums[i] + 1 == nums[i + 1]){
                right = i + 1;
            } else{
                string s ;
                if(left == right){
                    s = to_string(nums[left]);
                } else{
                    s = to_string(nums[left]) + "->" + to_string(nums[right]);
                }
                res.emplace_back(s);
                left = right = i + 1;
            }
        }
        return res;
    }

    int divide(int dividend, int divisor) {
            if(dividend == INT_MIN){
                if(divisor == 1){
                    return INT_MIN;
                }
                if(divisor == -1){
                    return INT_MAX;
                }
            }
            if(divisor == INT_MIN){
                return dividend == INT_MIN ? 1 : 0;
            }
            if(dividend == 0){
                return 0;
            }

            bool rev = false;
            if(dividend > 0){
                dividend = - dividend;
                rev = !rev;
            }
            if(divisor > 0){
                divisor = -divisor;
                rev = !rev;
            }

            auto quickAdd = [](int y, int z, int x){
                int result = 0, add = y;
                while (z){
                    if(z & 1){
                        if(result < x - add){
                            return false;
                        }
                        result += add;
                    }
                    if(z != 1){
                        if(add < x - add){
                            return false;
                        }
                        add +=add;
                    }
                    z >>= 1;
                }
                return true;
            };

            int left = 1, right = INT_MAX, ans = 0;
            while (left <= right){
                int mid = left + ((right - left) >> 1);
                bool check = quickAdd(divisor, mid, dividend);
                if(check){
                    ans = mid;
                    if(mid == INT_MAX){
                        break;
                    }
                    left = mid + 1;
                } else{
                    right = mid - 1;
                }
            }
        return rev ? -ans : ans;
    }

    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        vector<int> flag(n, 0); // 初始化一个大小为n的vector，初始值都为0
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            flag[nums[i] - 1] = 1;
        }
        for (int i = 0; i < n; ++i) {
            if (flag[i] != 1)res.push_back(i + 1);
        }
        return res;
    }
    int missingNumber(vector<int>& nums) {
            int n = nums.size();
            int res = 0;
        for (int i = 0; i <= n; ++i) {
            res = res ^ i;
        }
        for (int i = 0; i < n; ++i) {
            res = res ^ nums[i];
        }
        return res;
    }


    void perm(vector<vector<int>>& res,vector<int>& nums,vector<int>& flag,vector<int>& tmp,int& count){
            int n = nums.size();
            if(count == n){
                res.push_back(tmp);
                return;
            }
        for (int i = 0; i < n; ++i) {
            if (flag[i] == 0){
                if (i > 0 && nums[i] == nums[i - 1] && flag[i - 1] == 0)continue;
                tmp.push_back(nums[i]);
                flag[i] = 1;
                count++;
                perm(res,nums,flag,tmp,count);
                count--;
                flag[i] = 0;
                tmp.pop_back();
            }
        }
        }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
            int n = nums.size();
        vector<vector<int>> res;
        vector<int> flag(n,0);
        vector<int>;
        int count = 0;
        sort(nums.begin(),nums.end());
        perm(res,nums,flag,tmp,count);
        return res;
    }
        int thirdMax(vector<int>& nums) {
            int n = nums.size();
            if(n < 3){
                if (n == 1)return nums[0];
                return max(nums[0], nums[1]);
            }
            stack<int> S1,S2;
            S1.push(nums[0]);
            for (int i = 1; i < n; ++i) {
                if (S1.top() == nums[i])continue;
                if (S1.top() < nums[i] || S1.size() < 3){
                    while (!S1.empty() && S1.top() < nums[i]){
                        S2.push(S1.top());
                        S1.pop();
                    }
                    S1.push(nums[i]);
                    while (S1.size() < 3 && !S2.empty()){
                        S1.push(S2.top());
                        S2.pop();
                    }
                    while (!S2.empty())S2.pop();
                }
            }
            return S1.top();
    }
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if (matrix.empty())return res;
        int u = 0, d = matrix.size() - 1, l = 0, r = matrix[0].size() - 1;

        while (true){
            for (int i = l; i <= r; ++i) {
                res.push_back(matrix[u][i]);
            }
            if (++u > d)break;
            for (int i = u; i <= d; ++i) {
                res.push_back(matrix[i][r]);
            }
            if (--r < l)break;
            for (int i = r; i >= l; --i) {
                res.push_back(matrix[d][i]);
            }
            if (--d < u)break;
            for (int i = d; i >= u; --i) {
                res.push_back(matrix[i][l]);
            }
            if (++l > r)break;
        }
        return res;
    }
#pragma endregion

};


int main() {
    Solution solution;
    string s = "3[a]2[bc]";
    vector<int> nums = {1,2,3};
//    solution.permute(nums);
    return 0;
}
