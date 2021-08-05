//らせん階段
// カブト虫
// 廃墟の街
// イチジクのタルト
// カブト虫
//ドロローサへの道
// カブト虫
// 特異点
// ジョット
// エンジェル
// 紫陽花
// カブト虫
// 特異点
// 秘密の皇帝


#include <bits/stdc++.h>
#include<cmath>
using namespace std;
// #include<boost/multiprecision/cpp_int.hpp>
// using namespace boost::multiprecision;
using Graph = vector<vector<int>>;
#define ll long long
#define vvi(V,H,W) vector<vector<int>> (V)((H),vector<int>(W));
#define vvl(V,H,W) vector<vector<ll>> (V)((H),vector<ll>(W));
#define vvs(V,H,W) vector<vector<string>> (V)((H),vector<string>(W));
#define vvc(V,H,W) vector<vector<char>> (V)((H),vector<char>(W));
#define irep(n) for (int i=0; i < (n); ++i)
#define irepf1(n) for (int i=1; i <= (n); ++i)
#define jrep(n) for (int j=0; j < (n); ++j)
#define jrepf1(n) for (int j=1; j <= (n); ++j)
#define krep(n) for (int k=0; k < (n); ++k)
#define krepf1(n) for (int k=1; k <= (n); ++k)
#define REP(i,s,e) for (int (i)=(s); (i)<(e);(i)++)
#define PI 3.14159265358979323846264338327950288
#define mod 1000000007
#define eps 0.00000001
#define Find(V,X) find(V.begin(),V.end(),X)
#define Sort(V) sort((V).begin(),(V).end())
#define Reverse(V) reverse((V).begin(),(V).end())
#define Greater(V) sort((V).begin(),(V).end(),greater<int>())
#define cmin(ans,A) (ans)=min((ans),(A))
#define cmax(ans,A) (ans)=max((ans),(A))
#define AUTO(x,V) for (auto (x):(V))
#define int long long
//fixed << setprecision(10) <<
Graph makeGraph(int N, int V) {
  Graph G(N);
  irep (V) {
    int A,B;
    cin >>A>>B;
    A--;B--;
    G[A].push_back(B);
    G[B].push_back(A);
  }
  return G;
}

signed main() {
  
}