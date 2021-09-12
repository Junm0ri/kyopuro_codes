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
#define ld long double
#define pi pair<int,int>
#define PQ priority_queue<int>
#define PQG priority_queue<int, vector<int>, greater<int>>
#define vi vector<int> 
#define vld vector<long double> 
#define vs vector<string> 
#define vp vector<pi>
#define vvi(V,H,W) vector<vector<int>> (V)((H),vector<int>(W));
#define vvl(V,H,W) vector<vector<ll>> (V)((H),vector<ll>(W));
#define vvld(V,H,W) vector<vector<long double>> (V)((H),vector<long double>(W));
#define vvs(V,H,W) vector<vector<string>> (V)((H),vector<string>(W));
#define vvc(V,H,W) vector<vector<char>> (V)((H),vector<char>(W));
#define irep(n) for (int i=0; i < (n); ++i)
#define irepf1(n) for (int i=1; i <= (n); ++i)
#define jrep(n) for (int j=0; j < (n); ++j)
#define jrepf1(n) for (int j=1; j <= (n); ++j)
#define krep(n) for (int k=0; k < (n); ++k)
#define krepf1(n) for (int k=1; k <= (n); ++k)
#define REP(i,s,e) for (int (i)=(s); (i)<(e);(i)++)
#define PER(i,s,e) for (int (i)=(s); (i)>=(e);(i)--)
#define PI 3.14159265358979323846264338327950288
#define Banpei 1000000000 //問題毎に設定
#define Max_V 100000
#define mod7 1000000007
#define eps 0.00000001
#define ALL(V,A) ((V).begin(),(V).end(),(A))
#define Find(V,X) find(V.begin(),V.end(),X)
#define Lbound(V,X) *lower_bound((V).begin(),(V).end(),(X))
#define LboundP(V,X) lower_bound((V).begin(),(V).end(),(X))-(V).begin();
#define Ubound(V,X) *upper_bound((V).begin(),(V).end(),(X))
#define UboundP(V,X) upper_bound((V).begin(),(V).end(),(X))-(V).begin();
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

void solve() {
  
}

signed main() {
  cin.tie(0);
  ios::sync_with_stdio(false);
  int t;
  cin>>t;
  irep(t) solve();
}