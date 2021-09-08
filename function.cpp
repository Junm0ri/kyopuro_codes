//関数
bool isCrossing(int a,int b, int c, int d) { //閉区間[a,b],[c,d]が共通部分を持つか判定
  if (max(a,c)<=min(b,d) return 1;
  else return 0;
}
vector<long double> VRot(long double x, long double y, long double rad) {//ベクトルを回転する関数(引数はx,y,θ(rad))
  long double Sin=sin(rad);
  long double Cos=cos(rad);
  return {x*Cos-y*Sin,y*Cos+x*Sin};
}
int findSumOfDigits(int n){ //各桁の和を求める関数
  int sum =0;
  while(n > 0){
    sum += n % 10;
    n /= 10;
  }
    return sum;
}
bool is_palindrome(string s){ //回文判定
    string t(s.rbegin(),s.rend());
    return s == t;
}
string to_oct(int n){ //10進数から8進数（文字列）へ
  string s;
  while(n){
    s = to_string(n%8) + s;
    n /= 8;
  }
  return s;
}
string to_bin(int n){ //10進数から2進数（文字列）へ
  string s;
  while(n){
    s = to_string(n%2) + s;
    n /= 2;
  }
  return s;
}
int binary(int bina){ //10進数から2進数（数字列）へ
    int ans = 0;
    for (int i = 0; bina>0 ; i++)
    {
        ans = ans+(bina%2)*pow(10,i);
        bina = bina/2;
    }
    return ans;
} 
ll btod(bitset B) { //bitsetを10進数へ(未完成)
//   ll ret=0,stock=1;
//   while (B) {
//     ret+=stock*(B&1);
//     stock*=2;
//     B>>1;
//   }
//   return ret;
// }
int ctoi(const char c){ //character⇔int変換
  if('0' <= c && c <= '9') return (c-'0');
  return -1;
}
int isInt(double n) { //intか判定
}
bool isPrime(ll N) { //素数判定
  if (N<=1) return false;
  for (ll i=2; i*i<=N;i++) {
    if (N%i==0) return false;
  }
  return true;
}
int Keta(ll N) { //桁を求める
  int ret=1;
  while(1) {
    N/=10;
    if (N==0) return ret;
    ret++;
  }
}
int powN(ll N) { //Nが2の何乗かを求める。
  int ret=1;
  while(1) {
    N/=2;
    if (N==0) return ret;
    ret++;
  }
}
int modpow(ll n, ll p, ll m) {
    long long ans = 1;
    while (p > 0) {
        if (p & 1) {
            ans = ans * n % m;
        }
        p = p >> 1;
        n = n * n % m;
    }
    return ans;
}
ll factorial(ll n) { //階乗
    if (n > 0) {
        return n * factorial(n - 1);
    } else {
        return 1;
    }
}
vector<int> Erato(int N) { //エラトステネスのふるい
  vector<int> ret;
  vector<int> flag(N,1);
  flag[0]=0;
  flag[1]=0;
  for (int i=2;i*i<=N;i++) {
    if (flag[i]) {
      for (int j=0;i*(j+2)<N;j++) {
        flag[i*(j+2)]=0;
      }
    }
  }
  irep (N) if (flag[i]) ret.push_back(i);
  return ret;
}
map<ll,int> predisposition(ll n){ //素因数分解
  ll x = n;
  map<ll,int> ret;
  //cout << n << ":";
  for(ll i = 2; i * i <= x; i++){
    while(n % i== 0) {
      ret[i]++;
      n /= i;
    }
  }
  if (n!=1) ret[n]=1;
  return ret;
}
 {//高速素因数分解(メモリの都合上,10^8程度までのみ,main関数内でpreprocess()を宣言してからpredispositionを呼び出す

// https://drken1215.hatenablog.com/entry/2018/09/24/194500
const int MAX = 15001000;
bool IsPrime[MAX];
int MinFactor[MAX];
vector<int> preprocess(int n = MAX) {
    vector<int> res;
    for (int i = 0; i < n; ++i) IsPrime[i] = true, MinFactor[i] = -1;
    IsPrime[0] = false; IsPrime[1] = false; 
    MinFactor[0] = 0; MinFactor[1] = 1;
    for (int i = 2; i < n; ++i) {
        if (IsPrime[i]) {
            MinFactor[i] = i;
            res.push_back(i);
            for (int j = i*2; j < n; j += i) {
                IsPrime[j] = false;
                if (MinFactor[j] == -1) MinFactor[j] = i;
            }
        }
    }
    return res;
}
map<ll,int> fastPD(ll N) {
  map<ll,int> ret;
  while (N!=1) {
    ret[MinFactor[N]]++;
    N/=MinFactor[N];
  }
  return ret;
}

//使用例
int main() {
  ll N=109309;
  vector<ll> V(10000,N);
  preprocess();
  map M=fastPD(N);
  for (auto x:M) cout<<x.first<<" "<<x.second<<endl;
}
}
vector<vector<long long>> comb(int n, int r) { //組み合わせ（入力：comb(n,k) 返り値:v[n][k]で取得可能）
  vector<vector<long long>> v(n + 1,vector<long long>(n + 1, 0));
  for (int i = 0; i < (int)v.size(); i++) {
    v[i][0] = 1;
    v[i][i] = 1;
  }
  for (int j = 1; j < (int)v.size(); j++) {
    for (int k = 1; k < j; k++) {
      v[j][k] = (v[j - 1][k - 1] + v[j - 1][k]);
    }
  }
  return v;
}
vector<vector<long long>> modComb(int n, int r, int M) { //組み合わせ（入力：comb(n,k,mod) 返り値:v[n][k]で取得可能）
  vector<vector<long long>> v(n + 1,vector<long long>(n + 1, 0));
  for (int i = 0; i < (int)v.size(); i++) {
    v[i][0] = 1;
    v[i][i] = 1;
  }
  for (int j = 1; j < (int)v.size(); j++) {
    for (int k = 1; k < j; k++) {
      v[j][k] = (v[j - 1][k - 1] + v[j - 1][k])%M;
    }
  }
  return v;
}
// 使用例
// vector<vector<ll>> V=comb(100,4);
// cout<<V[100][2]
// cout<<comb(19900,2)[19900][2]<<endl;
string ID(int k, int n) { //桁つめ（例:Fun(6,123) 123→000123）
  string ret;
  string S;
  S=to_string(n);
  int x=n;
  while (x) {
    x/=10;
    k--;
  }
  irep (k) ret=ret+"0";
  ret=ret+S;
  return ret;
}
int Log(int a,int b,int n) { //a*b^x>=nとなるxを求めます
  int ret=0;
  while (n>a) {
    a*=b;
    ret++;
  }
  return ret;
}
ll modfact(ll A, int Mod) { //modを取りながらA!を求める
  if (A==1) return 1;
  else return (A%mod*(modfact(A-1,Mod)%mod))%Mod;
}
vector<int> enum_div(int n){ //自分以外の約数全列挙
    vector<int> ret;
    if (n==1) return ret;
    for(int i = 1 ; i*i <= n ; ++i){
        if(n%i == 0){
            ret.push_back(i);
            if(i != 1 && i*i != n){
                ret.push_back(n/i);
            }
        }
    }
    return ret;
}
int stringcount(string s, char c) { //文字数カウント
    return count(s.cbegin(), s.cend(), c);
}
void dfs_TreeSon (int pos,int pre) {//グローバル変数dpに自身を含む子の個数を格納する
  dp[pos]=1;
  for (int i:G[pos]) {
    if (i!=pre) {
      dfs(i,pos);
      dp[pos] +=dp [i];
    }
  }
}
#include<boost/multiprecision/cpp_int.hpp> 
//C++で多倍長整数を扱える（メモリと計算量に注意）
// 上のライブラリをincludeし、宣言をcpp_int Nとするだけで多倍長整数を使える


void mods() { //mod演算各種
/**
 * Return the Modulo result after subtraction.
 *
 * When the subtraction result becomes negative, add modulo value in order to be always positive.

 * eg. modMinus(5, 1, 10) -> 4(=(5-3)%10) 
 * eg. modMinus(1, 5, 10) -> 6(=(10+(3-5))%10) 
 */
template <typename T> //どんな型でもOK(だが、a,b,modは同じ型でなければならない。(ll)aなどで型変換すれば使用可)
T modMinus(T a, T b, T mod) {
  a %= mod;
  b %= mod;
  return (a>b)? (a-b)%mod: (mod+a-b)%mod;
}

/**
 * Return the Modulo result after power calculation.
 *
 * Implemented by the division-and-conquer method which is fast algorithm.
 * eg. modPow(2, 3, 10) -> 8(=(2*2*2)%10) 
 * eg. modPow(2, 4, 10) -> 6(=(2*2*2*2)%10) 
 */
template <typename T>
T modPow(T base, T e, T mod) {
  if(e == 0) return 1;
  if(e == 1) return base%mod;
  if(e%(T)2 == 1) return (base * modPow(base, e-(T)1, mod)) %mod;

  T tmp = modPow(base, e/(T)2, mod);
  return (tmp * tmp) % mod;
}

/**
 * Return the Modulo result after division.
 *
 * The dividing value must be a prime number because the inverse element is used.
 * In other words, this function uses Fermat's little theorem.
 */
template <typename T>
T primeModDiv(T num, T den, T primeMod){
  return ( num*modPow(den, primeMod-(T)2, primeMod) )%primeMod;
}

/**
 * Return the Modulo of binomial coefficients (Value of n-C-k %mod).
 *
 * The dividing value must be a prime number because the inverse element is used.
 * In other words, this function uses Fermat's little theorem.
 */
template <typename T>
T modComb(T n, T k, T mod) {
  T num = (T)1; //numerator
  T den = (T)1; //denominator
  for (T i = 1; i <= k; i++) {
    num *= (n-i+(T)1);
    num %= mod;
    den *= i;
    den %= mod;
  }
  return primeModDiv(num, den, mod);
}
}

// クラス
void DSU() {//互いに素な木集合 O(log n)
#include<iostream>
#include<vector>
class DisjointSet {//互いに素な木集合 O(log n)
  public:
    vector<int> rank,p;

    DisjointSet() {}
    DisjointSet(int size) {
      rank.resize(size,0);
      p.resize(size,0);
      for(int i=0;i<size;i++) makeSet(i);
    }

  void makeSet(int x) {
    p[x] =x;
    rank[x] =0;
  }
  bool same(int x, int y) {
    return findSet(x) ==findSet(y);
  }

  void unite(int x,int y) {
    link(findSet(x), findSet(y));
  }

  void link(int x,int y) {
    if (rank[x] > rank[y]) {
      p[y] =x;
    }
    else {
      p[x] =y;
      if (rank[x]==rank[y]) {
        rank[y]++;
      }
    }
  }

  int findSet(int x) {
    if (x!=p[x]) {
      p[x] =findSet(p[x]);
    }
    return p[x];
  }
    
};
使用例 {
DisjointSet ds=DisjointSet(n); //0~nまでの木を作成（全ての要素は互いに素である）
ds.unite(a,b) //aとbを結合
ds.same(a,b) //aとbが同じ木に属するかを判定（返り値:bool）

int main() {
  int n,q;
  cin >>n>>q;
  int t,a,b;
  DisjointSet ds=DisjointSet(n);
  irep (q) {
    cin >>t>>a >>b;
    if (t==0) ds.unite(a,b);
    else if (t==1) {
      if (ds.same(a,b)) cout<<1<<endl;
      else cout<<0<<endl;
    }
  }
}
}
/* UnionFind：素集合系管理の構造体(union by size)
    isSame(x, y): x と y が同じ集合にいるか。 計算量はならし O(α(n))
    unite(x, y): x と y を同じ集合にする。計算量はならし O(α(n))
    treeSize(x): x を含む集合の要素数。
*/
struct UnionFind { //0オリジンに修正して使用
    vector<int> size, parents;
    UnionFind() {}
    UnionFind(int n) {  // make n trees.
        size.resize(n, 0);
        parents.resize(n, 0);
        for (int i = 0; i < n; i++) {
            makeTree(i);
        }
    }
    void makeTree(int x) {
        parents[x] = x;  // the parent of x is x
        size[x] = 1;
    }
    bool isSame(int x, int y) { return findRoot(x) == findRoot(y); }
    bool unite(int x, int y) {
        x = findRoot(x);
        y = findRoot(y);
        if (x == y) return false;
        if (size[x] > size[y]) {
            parents[y] = x;
            size[x] += size[y];
        } else {
            parents[x] = y;
            size[y] += size[x];
        }
        return true;
    }
    int findRoot(int x) {
        if (x != parents[x]) {
            parents[x] = findRoot(parents[x]);
        }
        return parents[x];
    }
    int treeSize(int x) { return size[findRoot(x)]; }
};
}
void BFS{ //幅優先探索
// 各座標に格納したい情報を構造体にする
// 今回はX座標,Y座標,深さ(距離)を記述している
struct Corr {
    int x;
    int y;
    int depth;
};
queue<Corr> q;
int bfs(vector<vector<int>> grid) {
    // 既に探索の場所を1，探索していなかったら0を格納する配列
    vector<vector<int>> ispassed(grid.size(), vector<int>(grid[0].size(), false));
    // このような記述をしておくと，この後のfor文が綺麗にかける
    int dx[8] = {1, 0, -1, 0};
    int dy[8] = {0, 1, 0, -1};
    while(!q.empty()) {
        Corr now = q.front();q.pop();
        /*
            今いる座標は(x,y)=(now.x, now.y)で，深さ(距離)はnow.depthである
            ここで，今いる座標がゴール(探索対象)なのか判定する
        */
        for(int i = 0; i < 4; i++) {
            int nextx = now.x + dx[i];
            int nexty = now.y + dy[i];

            // 次に探索する場所のX座標がはみ出した時
            if(nextx >= grid[0].size()) continue;
            if(nextx < 0) continue;

            // 次に探索する場所のY座標がはみ出した時
            if(nexty >= grid.size()) continue;
            if(nexty < 0) continue;

            // 次に探索する場所が既に探索済みの場合
            if(ispassed[nexty][nextx]) continue;

            ispassed[nexty][nextx] = true;
            Corr next = {nextx, nexty, now.depth+1};
            q.push(next);
        }
    }
}

// 迷路問題テンプレ回答
// https://atcoder.jp/contests/abc007/submissions/21788801
int main(void){
  int R, C, sy, sx, gy, gx; //順番に幅、高さ、スタート（x,y）、ゴール(x,y)
 
  cin >> R >> C >> sy >> sx >> gy >> gx;
  
  // 0オリジンへ
  sy--;
  sx--;
  gy--;
  gx--;
  
  //番兵
  const int INF = 1001001001;
 
  //迷路の情報と回答用配列（スタートからの距離を格納）  
  vector<string> c(R);
  vector<vector<int>> ans(R, vector<int>(C, INF));
 
  for (int i = 0; i < R; ++i) cin >> c[i];
 
  queue<int> qy, qx;
  qy.push(sy);
  qx.push(sx);
 
  ans[sy][sx] = 0;
 
  int dy[] = {-1, 0, 1, 0};
  int dx[] = {0, 1, 0, -1};
 
  while (!qy.empty()) {
    int nowy = qy.front(); qy.pop();
    int nowx = qx.front(); qx.pop();
 
    for (int i = 0; i < 4; ++i) {
      int nexty = nowy + dy[i];
      int nextx = nowx + dx[i];
 
      if (nexty < 0 || nexty >= R ||
          nextx < 0 || nextx >= C ||
          ans[nexty][nextx] != INF || c[nexty][nextx] == '#') continue;
      else {
        ans[nexty][nextx] = ans[nowy][nowx] + 1;
        qy.push(nexty);
        qx.push(nextx);
      }
    }
  }
 
  cout << ans[gy][gx] << endl;
 
  return 0;
}

}
void Tentousu{ // 転倒数計算(BITによる)
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
// 1-indexedなので注意。
struct BIT {
 private:
  vector<int> bit;
  int N;

 public:
  BIT(int size) {
    N = size;
    bit.resize(N + 1);
  }

  // 一点更新です
  void add(int a, int w) {
    for (int x = a; x <= N; x += x & -x) bit[x] += w;
  }

  // 1~Nまでの和を求める。
  int sum(int a) {
    int ret = 0;
    for (int x = a; x > 0; x -= x & -x) ret += bit[x];
    return ret;
  }
};
// ====================================================================
int main() {
  int n;
  cin >> n;
  vector<int> v(n);
  for (int i = 0; i < n; i++) { cin >> v[i]; }

  ll ans = 0;
  BIT b(n);  // これまでの数字がどんな風になってるのかをメモる為のBIT
  for (int i = 0; i < n; i++) {
    ans += i - b.sum(v[i]); // BITの総和 - 自分より左側 ＝ 自分より右側
    b.add(v[i], 1); // 自分の位置に1を足す(ダジャレではないです)
  }

  cout << ans << endl;
}
}
void segT() { //セグメント木
/* RMQ：[0,n-1] について、区間ごとの最小値を管理する構造体
    update(a,b,x): 区間[a,b) の要素を x に更新。O(log(n))
    query(a,b): [a,b) での最小の要素を取得。O(log(n))
*/
template <typename T>
struct RMQ {
    const T INF = numeric_limits<T>::max();
    int n;
    vector<T> dat, lazy;
    RMQ(int n_) : n(), dat(n_ * 4, INF), lazy(n_ * 4, INF) {
        int x = 1;
        while (n_ > x) x *= 2;
        n = x;
    }
 
    void set(int i, T x) { dat[i + n - 1] = x; }
    void build() {
    for (int k = n - 2; k >= 0; k--){
        dat[k] = min(dat[2 * k + 1], dat[2 * k + 2]);
    }
}
    /* lazy eval */
    void eval(int k) {
        if (lazy[k] == INF) return;  // 更新するものが無ければ終了
        if (k < n - 1) {             // 葉でなければ子に伝搬
            lazy[k * 2 + 1] = lazy[k];
            lazy[k * 2 + 2] = lazy[k];
        }
        // 自身を更新
        dat[k] = lazy[k];
        lazy[k] = INF;
    }
 
    void update(int a, int b, T x, int k, int l, int r) {
        eval(k);
        if (a <= l && r <= b) {  // 完全に内側の時
            lazy[k] = x;
            eval(k);
        } else if (a < r && l < b) {                     // 一部区間が被る時
            update(a, b, x, k * 2 + 1, l, (l + r) / 2);  // 左の子
            update(a, b, x, k * 2 + 2, (l + r) / 2, r);  // 右の子
            dat[k] = min(dat[k * 2 + 1], dat[k * 2 + 2]);
        }
    }
    void update(int a, int b, T x) { update(a, b, x, 0, 0, n); }
 
    T query_sub(int a, int b, int k, int l, int r) {
        eval(k);
        if (r <= a || b <= l) {  // 完全に外側の時
            return INF;
        } else if (a <= l && r <= b) {  // 完全に内側の時
            return dat[k];
        } else {  // 一部区間が被る時
            T vl = query_sub(a, b, k * 2 + 1, l, (l + r) / 2);
            T vr = query_sub(a, b, k * 2 + 2, (l + r) / 2, r);
            return min(vl, vr);
        }
    }
    T query(int a, int b) { return query_sub(a, b, 0, 0, n); }
 
    /* debug */
    inline T operator[](int a) { return query(a, a + 1); }
    void print() {
        for (int i = 0; i < 2 * n - 1; ++i) {
            cout << (*this)[i];
            if (i != n) cout << ",";
        }
        cout << endl;
    }
};

//使用例（典型90 29）https://atcoder.jp/contests/typical90/tasks/typical90_ac
  signed main() {
    int W,N;
    cin >>W>>N;
    RMQ<int> A(W);
  
    irep (W) A.set(i,0);
    A.build();
 
    irep(N) {
    int L,R;
    cin >>L >>R;
    int X=A.query(L-1,R)-1;
    cout<<-1*X<<endl;
    A.update(L-1,R,X);
    }
  }
}
void Daikusutora() { //ダイクストラ法
    struct Edge {
      long long to;
      long long cost;
  };
  using Graph = vector<vector<Edge>>;
  using P = pair<long, int>;
  const long long INF = 100000000000000000;
  /* dijkstra(G,s,dis)
      入力：グラフ G, 開始点 s, 距離を格納する dis
      計算量：O(|E|log|V|)
      副作用：dis が書き換えられる
  */
  void dijkstra(const Graph &G, int s, vector<long long> &dis) {
    int N = G.size();
    dis.resize(N, INF);
    priority_queue<P, vector<P>, greater<P>> pq;  // 「仮の最短距離, 頂点」が小さい順に並ぶ
    dis[s] = 0;
    pq.emplace(dis[s], s);
    while (!pq.empty()) {
        P p = pq.top();
        pq.pop();
        int v = p.second;
        if (dis[v] < p.first) {  // 最短距離で無ければ無視
            continue;
        }
        for (auto &e : G[v]) {
            if (dis[e.to] > dis[v] + e.cost) {  // 最短距離候補なら priority_queue に追加
                dis[e.to] = dis[v] + e.cost;
                pq.emplace(dis[e.to], e.to);
            }
        }
    }
  }


  //使用例
  struct Edge {
    long long to;
    long long cost;
};
using Graph = vector<vector<Edge>>;
using P = pair<long, int>;
const long long INF = 100000000000000000;
/* dijkstra(G,s,dis)
    入力：グラフ G, 開始点 s, 距離を格納する dis
    計算量：O(|E|log|V|)
    副作用：dis が書き換えられる
*/
void dijkstra(const Graph &G, int s, vector<long long> &dis) {
    int N = G.size();
    dis.resize(N, INF);
    priority_queue<P, vector<P>, greater<P>> pq;  // 「仮の最短距離, 頂点」が小さい順に並ぶ
    dis[s] = 0;
    pq.emplace(dis[s], s);
    while (!pq.empty()) {
        P p = pq.top();
        pq.pop();
        int v = p.second;
        if (dis[v] < p.first) {  // 最短距離で無ければ無視
            continue;
        }
        for (auto &e : G[v]) {
            if (dis[e.to] > dis[v] + e.cost) {  // 最短距離候補なら priority_queue に追加
                dis[e.to] = dis[v] + e.cost;
                pq.emplace(dis[e.to], e.to);
            }
        }
    }
}

signed main() {
  int N;
  cin >>N;
  
  Graph G(N);
  irep (N-1) {
    int A,B,C;
    cin >>A>>B>>C;
    A--;B--;
    Edge D={B,C};
    Edge E={A,C};
    G[A].push_back(D);
    G[B].push_back(E);
  }
  
  int Q,K;
  cin >>Q>>K;
  K--;
  vector<long long> ans;
  dijkstra(G,K,ans);
  
  irep (Q) {
    int A,B;
    cin >>A>>B;
    A--;B--;
    cout<<ans[A]+ans[B]<<endl;
  }
}

 //使用例2(経路復元) https://atcoder.jp/contests/ahc005/submissions/24826767
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
#include <random>
using namespace std;
// #include<boost/multiprecision/cpp_int.hpp>
// using namespace boost::multiprecision;
const long long INF = 1LL << 60;
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





struct Edge {
    long long to;
    long long cost;
};

using Graph = vector<vector<Edge>>;
using P = pair<long, int>;
vector<string> V;
set<pair<int,int>> S;
set<pair<int,int>> No;
Graph G;
int N,si,sj;
int di[4]={0,1,0,-1};
int dj[4]={1,0,-1,0};

void dijkstra(const Graph &G, int s, vector<long long> &dis, vector<int> &prev) {
    int N = G.size();
    dis.resize(N, INF);
    prev.resize(N, -1); // 初期化
    priority_queue<P, vector<P>, greater<P>> pq; 
    dis[s] = 0;
    pq.emplace(dis[s], s);
    while (!pq.empty()) {
        P p = pq.top();
        pq.pop();
        int v = p.second;
        if (dis[v] < p.first) {
            continue;
        }
        for (auto &e : G[v]) {
            if (dis[e.to] > dis[v] + e.cost) {
                dis[e.to] = dis[v] + e.cost;
                prev[e.to] = v; // 頂点 v を通って e.to にたどり着いた
                pq.emplace(dis[e.to], e.to);
            }
        }
    }
}

vector<int> get_path(const vector<int> &prev, int t) {
    vector<int> path;
    for (int cur = t; cur != -1; cur = prev[cur]) {
        path.push_back(cur);
    }
    reverse(path.begin(), path.end()); // 逆順なのでひっくり返す
    return path;
}

void see(int A,int B){ //見える範囲を確認する
  //cout<<6<<endl;
  S.erase({A,B});
  int i=A,j=B;
  //cout<<7<<endl;
  while (1) {
    i++;
    if (i>=N) break;
    if (V[i][j]=='#') break;
    S.erase({i,j});
  }
  i=A,j=B;
  while (1) {
    i--;
    if (i<0) break;
    if (V[i][j]=='#') break;
    S.erase({i,j});
  }
  i=A,j=B;
  while (1) {
    j++;
    if (j>=N) break;
    if (V[i][j]=='#') break;
    S.erase({i,j});
  }
  i=A,j=B;
  while (1) {
    j--;
    if (j<0) break;
    if (V[i][j]=='#') break;
    S.erase({i,j});
  }
  return;
}

void move(int fi,int fj,int ti,int tj) {
  vector<int> dis;
  vector<int> prev;
  dijkstra(G,fi*N+fj,dis,prev);
  vector<int> Path=get_path(prev,ti*N+tj);
  int PS=Path.size();
  
  irepf1(PS-1) {
    //cout<<Path[i]<<endl;
    int ci=Path[i]/N;
    int cj=Path[i]%N;
    see(ci,cj);
    if (Path[i]-Path[i-1]==1) cout<<'R';
    else if (Path[i]-Path[i-1]==-1) cout<<'L';
    else if (Path[i]-Path[i-1]==-N) cout<<'U';
    else if (Path[i]-Path[i-1]==N) cout<<'D';
  }
  //cout<<5<<endl;
  return;
}

signed main() {
  cin >>N >>si>>sj;
  V.resize(N);
  G.resize(N*N);
  vector<pair<int,int>> stock;
  irep(N) cin >>V[i];
  irep(N) { //グラフの作成
      jrep(N) {
          if (V[i][j]=='#') continue;
          krep(4) {
              int ni=i+di[k];
              int nj=j+dj[k];
              if (ni<0||nj<0||ni>=N||nj>=N||V[ni][nj]=='#') continue;
              Edge E={ni*N+nj,V[ni][nj]-'0'};
              G[i*N+j].push_back(E);
          }
      }
  }
  
  irep(N) { //setの要素作成
      jrep(N) if (V[i][j]!='#') {
          S.insert({i,j});
          stock.push_back({i,j});
      }
    else No.insert({i,j});
  }
  int Size=stock.size();
  
    // irep(N*N) { //グラフの確認
    //     cout<<i<<" ";
    //     for (auto x:G[i]) cout<<x.to<<" "<<x.cost<<" ";
    //     cout<<endl;
    // }

  int i=si,j=sj;
  //cout<<S.size()<<endl;
  /*while (1) {
    see(i,j);
    int ni=rand()%N;
    int nj=rand()%N;
    //cout<<ni<<" "<<nj<<endl;
    if (S.count({ni,nj})||No.count({ni,nj})) continue;
    move(i,j,ni,nj);
    i=ni;
    j=nj;
    if (S.size()==0) break;
  }*/
  bool flag=true;
  while (S.size()!=0) {
    int ni,nj;
    //cout<<1<<endl;
    for (auto x:S) {
      ni=x.first;
      nj=x.second;
      if (flag) break;
    }
    move(i,j,ni,nj);
    i=ni;
    j=nj;
    flag=!flag;
  }
  move(i,j,si,sj); //開始地点へ
}
}
void Kraskal() { //クラスカル法

// 素集合データ構造
struct UnionFind
{
  // par[i]：データiが属する木の親の番号。i == par[i]のとき、データiは木の根ノードである
  vector<int> par;
  // sizes[i]：根ノードiの木に含まれるデータの数。iが根ノードでない場合は無意味な値となる
  vector<int> sizes;

  UnionFind(int n) : par(n), sizes(n, 1) {
    // 最初は全てのデータiがグループiに存在するものとして初期化
    irep(n) par[i] = i;
  }

  // データxが属する木の根を得る
  int find(int x) {
    if (x == par[x]) return x;
    return par[x] = find(par[x]);  // 根を張り替えながら再帰的に根ノードを探す
  }

  // 2つのデータx, yが属する木をマージする
  void unite(int x, int y) {
    // データの根ノードを得る
    x = find(x);
    y = find(y);

    // 既に同じ木に属しているならマージしない
    if (x == y) return;

    // xの木がyの木より大きくなるようにする
    if (sizes[x] < sizes[y]) swap(x, y);

    // xがyの親になるように連結する
    par[y] = x;
    sizes[x] += sizes[y];
    // sizes[y] = 0;  // sizes[y]は無意味な値となるので0を入れておいてもよい
  }

  // 2つのデータx, yが属する木が同じならtrueを返す
  bool same(int x, int y) {
    return find(x) == find(y);
  }

  // データxが含まれる木の大きさを返す
  int size(int x) {
    return sizes[find(x)];
  }
};

  // 頂点a, bをつなぐコストcostの（無向）辺
struct Edge
{
  int a, b, cost;

  // コストの大小で順序定義
  bool operator<(const Edge& o) const {
    return cost < o.cost;
  }
};

// 頂点数と辺集合の組として定義したグラフ
struct Graph
{
  int n;  // 頂点数
  vector<Edge> es;  // 辺集合

  // クラスカル法で無向最小全域木のコストの和を計算する
  // グラフが非連結のときは最小全域森のコストの和となる
  int kruskal() {
    // コストが小さい順にソート
    sort(es.begin(), es.end());

    UnionFind uf(n);
    int min_cost = 0;

    irep(es.size()) {
      Edge& e = es[i];
      if (!uf.same(e.a, e.b)) {
        // 辺を追加しても閉路ができないなら、その辺を採用する
        min_cost += e.cost;
        uf.unite(e.a, e.b);
      }
    }

    return min_cost;
  }
};

//使用例
// 標準入力からグラフを読み込む
Graph input_graph() {
  Graph g;
  int m;
  cin >> g.n >> m;
  irep(m) {
    Edge e;
    cin >> e.a >> e.b >> e.cost;
    g.es.push_back(e);
  }
  return g;
}

int main()
{
  Graph g = input_graph();
  cout << "最小全域木のコスト: " << g.kruskal() << endl;
  return 0;
}
}
void TreeDiameter { //木の直径を求める
  struct Edge {
    int to;
    int cost;
};
using Graph = vector<vector<Edge>>;  // cost の型を long long に指定

/* tree_diamiter : dfs を用いて重み付き木 T の直径を求める
    計算量: O(N)
*/
pair<int, int> dfs(const Graph &G, int u, int par) {  // 最遠点間距離と最遠点を求める
    pair<int, int> ret = make_pair(0, u);
    for (auto e : G[u]) {
        if (e.to == par) continue;
        auto next = dfs(G, e.to, u);
        next.first += e.cost;
        ret = max(ret, next);
    }
    return ret;
}

int tree_diamiter(const Graph &G) {
    pair<int, int> p = dfs(G, 0, -1);
    pair<int, int> q = dfs(G, p.second, -1);
    return q.first;
}

//使用例
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
using namespace std;
// #include<boost/multiprecision/cpp_int.hpp>
// using namespace boost::multiprecision;
// using Graph = vector<vector<int>>;
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

 
struct Edge {
    int to;
    int cost;
};
using Graph = vector<vector<Edge>>;  // cost の型を long long に指定

/* tree_diamiter : dfs を用いて重み付き木 T の直径を求める
    計算量: O(N)
*/
pair<int, int> dfs(const Graph &G, int u, int par) {  // 最遠点間距離と最遠点を求める
    pair<int, int> ret = make_pair(0, u);
    for (auto e : G[u]) {
        if (e.to == par) continue;
        auto next = dfs(G, e.to, u);
        next.first += e.cost;
        ret = max(ret, next);
    }
    return ret;
}

int tree_diamiter(const Graph &G) {
    pair<int, int> p = dfs(G, 0, -1);
    pair<int, int> q = dfs(G, p.second, -1);
    return q.first;
}

signed main() {
  int N;
  cin >>N;
  Graph G(N);
  irep(N-1) {
    int A,B;
    cin >>A>>B;
    A--;B--;
    Edge EA={B,1};
    Edge EB={A,1};
    G[A].push_back(EA);
    G[B].push_back(EB);

  }
  cout<<tree_diamiter(G)+1<<endl;

}


}
void SCC{ //強連結成分分解
  struct SCC { //強連結成分分解
  
  int V;
  map<int,int> MP; //各強連結成分の要素数
  vector<int> vs; //帰りがけの順番
  bool used[100001]; //頂点が探索済みか否か
  int cmp[100001]; //強連結成分のトポロジカル順序
  int CSize=0; //強連結成分の数
  Graph G;
  Graph rG;

  SCC(Graph A, Graph rA) {
    G=A;
    rG=rA;
    V=G.size();
    memset(used,0,sizeof(used));
    vs.clear();
    for (int v=0; v<V;v++) if (!used[v]) dfs(v);
    memset(used,0,sizeof(used));
    for (int i=vs.size()-1;i>=0;i--) {
      if (!used[vs[i]]) rdfs(vs[i],CSize++);
    }
  }

  void dfs(int v) {
    used[v]=true;
    for (auto x:G[v]) {
      if (!used[x]) dfs(x);
    }
    vs.push_back(v);
    }

  void rdfs(int v, int k) {
    MP[k]++;
    used[v]=true;
    cmp[v]=k;
    for (auto x:rG[v]) if (!used[x]) rdfs(x,k);
    }

  };

  //使用例（典型90 21） https://atcoder.jp/contests/typical90/submissions/24791685

  signed main() {
  int N,M;
  cin >>N >>M;
  Graph G(N);
  Graph rG(N);
  irep (M) {
    int A,B;
    cin >>A>>B;
    A--;B--;
    G[A].push_back(B);
    rG[B].push_back(A);
  }

  SCC S=SCC(G,rG);


  int ans=0;
  for (auto x:S.MP) {
    ans+=x.second*(x.second-1)/2;
  }
  cout<<ans<<endl;
  
}

}
void MaxFlow{ //最大フロー
  struct Edge {
  int to,cap,rev;
};

//ヘッダーのMax_Vを調整して使用,Graphを作成し、max_flow(s,t)を呼び出すことで最大フローが返される
vector<vector<Edge>> G(Max_V);
bool used[Max_V];

void add_Edge(int from, int to, int cap) {
  G[from].push_back((Edge) {to,cap,G[to].size()});
  G[to].push_back((Edge) {from,0,G[from].size()-1});
}

int dfs(int v, int t, int f) {
  if (v==t) return f;
  used[v] =true;

  for (int i=0; i<G[v].size();i++) {
    Edge &e =G[v][i];
    if (!used[e.to] && e.cap>0) {
      int d =dfs(e.to, t ,min(f,e.cap));
      if (d>0) {
        e.cap -=d;
        G[e.to][e.rev].cap+=d;
        return d;
      }
    }
  }
  return 0;
}

int max_flow(int s, int t) {//s:始点,t:終点
  int flow=0;
  while(1) {
    memset(used,0,sizeof(used));
    int f =dfs(s,t,INF);
    if (f==0) return flow;
    flow +=f;
  }
}

　//使用例(ARC092 C):最大マッチングとして利用
　struct Edge {
  int to,cap,rev;
};

// using Graph =vector<vector<Edge>>;
vector<vector<Edge>> G(Max_V);
bool used[Max_V];

void add_Edge(int from, int to, int cap) {
  G[from].push_back((Edge) {to,cap,G[to].size()});
  G[to].push_back((Edge) {from,0,G[from].size()-1});
}

int dfs(int v, int t, int f) {
  if (v==t) return f;
  used[v] =true;

  for (int i=0; i<G[v].size();i++) {
    Edge &e =G[v][i];
    if (!used[e.to] && e.cap>0) {
      int d =dfs(e.to, t ,min(f,e.cap));
      if (d>0) {
        e.cap -=d;
        G[e.to][e.rev].cap+=d;
        return d;
      }
    }
  }
  return 0;
}

int max_flow(int s, int t) {//s:始点,t:終点
  int flow=0;
  while(1) {
    memset(used,0,sizeof(used));
    int f =dfs(s,t,INF);
    if (f==0) return flow;
    flow +=f;
  }
}

signed main() {
  int N;
  cin >>N;
  // G.resize(2*N+2);
  vector<P> A(N);
  vector<P> C(N);
  irep(N) cin >>A[i].first>>A[i].second;
  irep(N) cin >>C[i].first>>C[i].second;
  irepf1(N) add_Edge(0,i,1);
  irepf1(N) add_Edge(i+N,2*N+1,1);
  irep(N) {
    jrep(N) {
      if (A[i].first<C[j].first&&A[i].second<C[j].second) {
        add_Edge(i+1,j+1+N,1);
        //cout<<100000000000<<endl;
      }
    }
  }
  cout<<max_flow(0,2*N+1)<<endl;
}

}
void dynamicConnectivity{ //辺の削除が可能なDSU https://qiita.com/hotman78/items/78cd3aa50b05a57738d4
  template<typename T>
class dynamic_connectivity{
    class euler_tour_tree{
        public:
        struct node;
        using np=node*;
        using lint=long long;
        struct node{
            np ch[2]={nullptr,nullptr};
            np p=nullptr;
            int l,r,sz;
            T val=et,sum=et;
            bool exact;
            bool child_exact;
            bool edge_connected=0;
            bool child_edge_connected=0;
            node(){}
            node(int l,int r):l(l),r(r),sz(l==r),exact(l<r),child_exact(l<r){}
            bool is_root() {
                return !p;
            }
        };
        vector<unordered_map<int,np>>ptr;
        np get_node(int l,int r){
            if(ptr[l].find(r)==ptr[l].end())ptr[l][r]=new node(l,r);
            return ptr[l][r];
        }
        np root(np t){
            if(!t)return t;
            while(t->p)t=t->p;
            return t;
        }
        bool same(np s,np t){
            if(s)splay(s);
            if(t)splay(t);
            return root(s)==root(t);
        }
        np reroot(np t){
            auto s=split(t);
            return merge(s.second,s.first);
        }
        pair<np,np> split(np s){
            splay(s);
            np t=s->ch[0];
            if(t)t->p=nullptr;
            s->ch[0]=nullptr;
            return {t,update(s)};
        }
        pair<np,np> split2(np s){
            splay(s);
            np t=s->ch[0];
            np u=s->ch[1];
            if(t)t->p=nullptr;
            s->ch[0]=nullptr;
            if(u)u->p=nullptr;
            s->ch[1]=nullptr;
            return {t,u};
        }
        tuple<np,np,np> split(np s,np t){
            auto u=split2(s);
            if(same(u.first,t)){
                auto r=split2(t);
                return make_tuple(r.first,r.second,u.second);
            }else{
                auto r=split2(t);
                return make_tuple(u.first,r.first,r.second);
            }
        }
        template<typename First, typename... Rest>
        np merge(First s,Rest... t){
            return merge(s,merge(t...));
        }
        np merge(np s,np t){
            if(!s)return t;
            if(!t)return s;
            while(s->ch[1])s=s->ch[1];
            splay(s);
            s->ch[1]=t;
            if(t)t->p=s;
            return update(s);
        }
        int size(np t){return t?t->sz:0;}
        np update(np t){
            t->sum=et;
            if(t->ch[0])t->sum=fn(t->sum,t->ch[0]->sum);
            if(t->l==t->r)t->sum=fn(t->sum,t->val);
            if(t->ch[1])t->sum=fn(t->sum,t->ch[1]->sum);
            t->sz=size(t->ch[0])+(t->l==t->r)+size(t->ch[1]);
            t->child_edge_connected=(t->ch[0]?t->ch[0]->child_edge_connected:0)|(t->edge_connected)|(t->ch[1]?t->ch[1]->child_edge_connected:0);
            t->child_exact=(t->ch[0]?t->ch[0]->child_exact:0)|(t->exact)|(t->ch[1]?t->ch[1]->child_exact:0);
            return t;
        }
        void push(np t){
            //遅延評価予定
        }
        void rot(np t,bool b){
            np x=t->p,y=x->p;
            if((x->ch[1-b]=t->ch[b]))t->ch[b]->p=x;
            t->ch[b]=x,x->p=t;
            update(x);update(t);
            if((t->p=y)){
                if(y->ch[0]==x)y->ch[0]=t;
                if(y->ch[1]==x)y->ch[1]=t;
                update(y);
            }
        }
        void splay(np t){
            push(t);
            while(!t->is_root()){
                np q=t->p;
                if(q->is_root()){
                    push(q), push(t);
                    rot(t,q->ch[0]==t);
                }else{
                    np r=q->p;
                    push(r), push(q), push(t);
                    bool b=r->ch[0]==q;
                    if(q->ch[1-b]==t)rot(q,b),rot(t,b);
                    else rot(t,1-b),rot(t,b);
                }
            }
        }
        void debug(np t){
            if(!t)return;
            debug(t->ch[0]);
            cerr<<t->l<<"-"<<t->r<<" ";
            debug(t->ch[1]);
        }
        public:
        euler_tour_tree(){}
        euler_tour_tree(int sz){
            ptr.resize(sz);
            for(int i=0;i<sz;i++)ptr[i][i]=new node(i,i);
        }
        int size(int s){
            np t=get_node(s,s);
            splay(t);
            return t->sz;
        }
        bool same(int s,int t){
            return same(get_node(s,s),get_node(t,t));
        }
        void set_size(int sz){
            ptr.resize(sz);
            for(int i=0;i<sz;i++)ptr[i][i]=new node(i,i);
        }
        void update(int s,T x){
            np t=get_node(s,s);
            splay(t);
            t->val=fn(t->val,x);
            update(t);
        }
        void edge_update(int s,auto g){
            np t=get_node(s,s);
            splay(t);
            function<void(np)>dfs=[&](np t){
                assert(t);
                if(t->l<t->r&&t->exact){
                    splay(t);
                    t->exact=0;
                    update(t);
                    g(t->l,t->r);
                    return;
                }
                if(t->ch[0]&&t->ch[0]->child_exact)dfs(t->ch[0]);
                else dfs(t->ch[1]);
            };
            while(t&&t->child_exact){
                dfs(t);
                splay(t);
            }
        }
        bool try_reconnect(int s,auto f){
            np t=get_node(s,s);
            splay(t);
            function<bool(np)>dfs=[&](np t)->bool{
                assert(t);
                if(t->edge_connected){
                    splay(t);
                    return f(t->l);
                }
                if(t->ch[0]&&t->ch[0]->child_edge_connected)return dfs(t->ch[0]);
                else return dfs(t->ch[1]);
            };
            while(t->child_edge_connected){
                if(dfs(t))return 1;
                splay(t);
            }
            return 0;
        }
        void edge_connected_update(int s,bool b){
            np t=get_node(s,s);
            splay(t);
            t->edge_connected=b;
            update(t);
        }
        bool link(int l,int r){
            if(same(l,r))return 0;
            merge(reroot(get_node(l,l)),get_node(l,r),reroot(get_node(r,r)),get_node(r,l));
            return 1;
        }
        bool cut(int l,int r){
            if(ptr[l].find(r)==ptr[l].end())return 0;
            np s,t,u;
            tie(s,t,u)=split(get_node(l,r),get_node(r,l));
            merge(s,u);
            np p=ptr[l][r];
            np q=ptr[r][l];
            ptr[l].erase(r);
            ptr[r].erase(l);
            delete p;delete q;
            return 1;
        }
        T get_sum(int p,int v){
            cut(p,v);
            np t=get_node(v,v);
            splay(t);
            T res=t->sum;
            link(p,v);
            return res;
        }
        T get_sum(int s){
            np t=get_node(s,s);
            splay(t);
            return t->sum;
        }
    };
    int dep=1;
    vector<euler_tour_tree> ett;
    vector<vector<unordered_set<int>>>edges;
    int sz;
    public:
    dynamic_connectivity(int sz):sz(sz){
        ett.emplace_back(sz);
        edges.emplace_back(sz);
    }
    bool link(int s,int t){
        if(s==t)return 0;
        if(ett[0].link(s,t))return 1;
        edges[0][s].insert(t);
        edges[0][t].insert(s);
        if(edges[0][s].size()==1)ett[0].edge_connected_update(s,1);
        if(edges[0][t].size()==1)ett[0].edge_connected_update(t,1);
        return 0;
    }
    bool same(int s,int t){
        return ett[0].same(s,t);
    }
    int size(int s){
        return ett[0].size(s);
    }
    vector<int>get_vertex(int s){
        return ett[0].vertex_list(s);
    }
    void update(int s,T x){
        ett[0].update(s,x);
    }
    T get_sum(int s){
        return ett[0].get_sum(s);
    }
    bool cut(int s,int t){
        if(s==t)return 0;
        for(int i=0;i<dep;i++){
            edges[i][s].erase(t);
            edges[i][t].erase(s);
            if(edges[i][s].size()==0)ett[i].edge_connected_update(s,0);
            if(edges[i][t].size()==0)ett[i].edge_connected_update(t,0);
        }
        for(int i=dep-1;i>=0;i--){
            if(ett[i].cut(s,t)){
                if(dep-1==i){
                    dep++;
                    ett.emplace_back(sz);
                    edges.emplace_back(sz);
                }
                return !try_reconnect(s,t,i);
            }
        }
        return 0;
    }
    bool try_reconnect(int s,int t,int k){
        for(int i=0;i<k;i++){
            ett[i].cut(s,t);
        }
        for(int i=k;i>=0;i--){
            if(ett[i].size(s)>ett[i].size(t))swap(s,t);
            auto g=[&](int s,int t){ett[i+1].link(s,t);};
            ett[i].edge_update(s,g);
            auto f=[&](int x)->bool{
                for(auto itr=edges[i][x].begin();itr!=edges[i][x].end();){
                    auto y=*itr;
                    itr=edges[i][x].erase(itr);
                    edges[i][y].erase(x);
                    if(edges[i][x].size()==0)ett[i].edge_connected_update(x,0);
                    if(edges[i][y].size()==0)ett[i].edge_connected_update(y,0);
                    if(ett[i].same(x,y)){
                        edges[i+1][x].insert(y);
                        edges[i+1][y].insert(x);
                        if(edges[i+1][x].size()==1)ett[i+1].edge_connected_update(x,1);
                        if(edges[i+1][y].size()==1)ett[i+1].edge_connected_update(y,1);
                    }else{
                        for(int j=0;j<=i;j++){
                            ett[j].link(x,y);
                        }
                        return 1;
                    }
                }
                return 0;
            };
            if(ett[i].try_reconnect(s,f))return 1;
        }
        return 0;
    }
    constexpr static T et=T();
    constexpr static T fn(T s,T t){
        return s+t;
    }
};
  //使用例 ABC124D(TLE) https://atcoder.jp/contests/abc214/tasks/abc214_d

template<typename T>
class dynamic_connectivity{
    class euler_tour_tree{
        public:
        struct node;
        using np=node*;
        using lint=long long;
        struct node{
            np ch[2]={nullptr,nullptr};
            np p=nullptr;
            int l,r,sz;
            T val=et,sum=et;
            bool exact;
            bool child_exact;
            bool edge_connected=0;
            bool child_edge_connected=0;
            node(){}
            node(int l,int r):l(l),r(r),sz(l==r),exact(l<r),child_exact(l<r){}
            bool is_root() {
                return !p;
            }
        };
        vector<unordered_map<int,np>>ptr;
        np get_node(int l,int r){
            if(ptr[l].find(r)==ptr[l].end())ptr[l][r]=new node(l,r);
            return ptr[l][r];
        }
        np root(np t){
            if(!t)return t;
            while(t->p)t=t->p;
            return t;
        }
        bool same(np s,np t){
            if(s)splay(s);
            if(t)splay(t);
            return root(s)==root(t);
        }
        np reroot(np t){
            auto s=split(t);
            return merge(s.second,s.first);
        }
        pair<np,np> split(np s){
            splay(s);
            np t=s->ch[0];
            if(t)t->p=nullptr;
            s->ch[0]=nullptr;
            return {t,update(s)};
        }
        pair<np,np> split2(np s){
            splay(s);
            np t=s->ch[0];
            np u=s->ch[1];
            if(t)t->p=nullptr;
            s->ch[0]=nullptr;
            if(u)u->p=nullptr;
            s->ch[1]=nullptr;
            return {t,u};
        }
        tuple<np,np,np> split(np s,np t){
            auto u=split2(s);
            if(same(u.first,t)){
                auto r=split2(t);
                return make_tuple(r.first,r.second,u.second);
            }else{
                auto r=split2(t);
                return make_tuple(u.first,r.first,r.second);
            }
        }
        template<typename First, typename... Rest>
        np merge(First s,Rest... t){
            return merge(s,merge(t...));
        }
        np merge(np s,np t){
            if(!s)return t;
            if(!t)return s;
            while(s->ch[1])s=s->ch[1];
            splay(s);
            s->ch[1]=t;
            if(t)t->p=s;
            return update(s);
        }
        int size(np t){return t?t->sz:0;}
        np update(np t){
            t->sum=et;
            if(t->ch[0])t->sum=fn(t->sum,t->ch[0]->sum);
            if(t->l==t->r)t->sum=fn(t->sum,t->val);
            if(t->ch[1])t->sum=fn(t->sum,t->ch[1]->sum);
            t->sz=size(t->ch[0])+(t->l==t->r)+size(t->ch[1]);
            t->child_edge_connected=(t->ch[0]?t->ch[0]->child_edge_connected:0)|(t->edge_connected)|(t->ch[1]?t->ch[1]->child_edge_connected:0);
            t->child_exact=(t->ch[0]?t->ch[0]->child_exact:0)|(t->exact)|(t->ch[1]?t->ch[1]->child_exact:0);
            return t;
        }
        void push(np t){
            //遅延評価予定
        }
        void rot(np t,bool b){
            np x=t->p,y=x->p;
            if((x->ch[1-b]=t->ch[b]))t->ch[b]->p=x;
            t->ch[b]=x,x->p=t;
            update(x);update(t);
            if((t->p=y)){
                if(y->ch[0]==x)y->ch[0]=t;
                if(y->ch[1]==x)y->ch[1]=t;
                update(y);
            }
        }
        void splay(np t){
            push(t);
            while(!t->is_root()){
                np q=t->p;
                if(q->is_root()){
                    push(q), push(t);
                    rot(t,q->ch[0]==t);
                }else{
                    np r=q->p;
                    push(r), push(q), push(t);
                    bool b=r->ch[0]==q;
                    if(q->ch[1-b]==t)rot(q,b),rot(t,b);
                    else rot(t,1-b),rot(t,b);
                }
            }
        }
        void debug(np t){
            if(!t)return;
            debug(t->ch[0]);
            cerr<<t->l<<"-"<<t->r<<" ";
            debug(t->ch[1]);
        }
        public:
        euler_tour_tree(){}
        euler_tour_tree(int sz){
            ptr.resize(sz);
            for(int i=0;i<sz;i++)ptr[i][i]=new node(i,i);
        }
        int size(int s){
            np t=get_node(s,s);
            splay(t);
            return t->sz;
        }
        bool same(int s,int t){
            return same(get_node(s,s),get_node(t,t));
        }
        void set_size(int sz){
            ptr.resize(sz);
            for(int i=0;i<sz;i++)ptr[i][i]=new node(i,i);
        }
        void update(int s,T x){
            np t=get_node(s,s);
            splay(t);
            t->val=fn(t->val,x);
            update(t);
        }
        void edge_update(int s,auto g){
            np t=get_node(s,s);
            splay(t);
            function<void(np)>dfs=[&](np t){
                assert(t);
                if(t->l<t->r&&t->exact){
                    splay(t);
                    t->exact=0;
                    update(t);
                    g(t->l,t->r);
                    return;
                }
                if(t->ch[0]&&t->ch[0]->child_exact)dfs(t->ch[0]);
                else dfs(t->ch[1]);
            };
            while(t&&t->child_exact){
                dfs(t);
                splay(t);
            }
        }
        bool try_reconnect(int s,auto f){
            np t=get_node(s,s);
            splay(t);
            function<bool(np)>dfs=[&](np t)->bool{
                assert(t);
                if(t->edge_connected){
                    splay(t);
                    return f(t->l);
                }
                if(t->ch[0]&&t->ch[0]->child_edge_connected)return dfs(t->ch[0]);
                else return dfs(t->ch[1]);
            };
            while(t->child_edge_connected){
                if(dfs(t))return 1;
                splay(t);
            }
            return 0;
        }
        void edge_connected_update(int s,bool b){
            np t=get_node(s,s);
            splay(t);
            t->edge_connected=b;
            update(t);
        }
        bool link(int l,int r){
            if(same(l,r))return 0;
            merge(reroot(get_node(l,l)),get_node(l,r),reroot(get_node(r,r)),get_node(r,l));
            return 1;
        }
        bool cut(int l,int r){
            if(ptr[l].find(r)==ptr[l].end())return 0;
            np s,t,u;
            tie(s,t,u)=split(get_node(l,r),get_node(r,l));
            merge(s,u);
            np p=ptr[l][r];
            np q=ptr[r][l];
            ptr[l].erase(r);
            ptr[r].erase(l);
            delete p;delete q;
            return 1;
        }
        T get_sum(int p,int v){
            cut(p,v);
            np t=get_node(v,v);
            splay(t);
            T res=t->sum;
            link(p,v);
            return res;
        }
        T get_sum(int s){
            np t=get_node(s,s);
            splay(t);
            return t->sum;
        }
    };
    int dep=1;
    vector<euler_tour_tree> ett;
    vector<vector<unordered_set<int>>>edges;
    int sz;
    public:
    dynamic_connectivity(int sz):sz(sz){
        ett.emplace_back(sz);
        edges.emplace_back(sz);
    }
    bool link(int s,int t){
        if(s==t)return 0;
        if(ett[0].link(s,t))return 1;
        edges[0][s].insert(t);
        edges[0][t].insert(s);
        if(edges[0][s].size()==1)ett[0].edge_connected_update(s,1);
        if(edges[0][t].size()==1)ett[0].edge_connected_update(t,1);
        return 0;
    }
    bool same(int s,int t){
        return ett[0].same(s,t);
    }
    int size(int s){
        return ett[0].size(s);
    }
    vector<int>get_vertex(int s){
        return ett[0].vertex_list(s);
    }
    void update(int s,T x){
        ett[0].update(s,x);
    }
    T get_sum(int s){
        return ett[0].get_sum(s);
    }
    bool cut(int s,int t){
        if(s==t)return 0;
        for(int i=0;i<dep;i++){
            edges[i][s].erase(t);
            edges[i][t].erase(s);
            if(edges[i][s].size()==0)ett[i].edge_connected_update(s,0);
            if(edges[i][t].size()==0)ett[i].edge_connected_update(t,0);
        }
        for(int i=dep-1;i>=0;i--){
            if(ett[i].cut(s,t)){
                if(dep-1==i){
                    dep++;
                    ett.emplace_back(sz);
                    edges.emplace_back(sz);
                }
                return !try_reconnect(s,t,i);
            }
        }
        return 0;
    }
    bool try_reconnect(int s,int t,int k){
        for(int i=0;i<k;i++){
            ett[i].cut(s,t);
        }
        for(int i=k;i>=0;i--){
            if(ett[i].size(s)>ett[i].size(t))swap(s,t);
            auto g=[&](int s,int t){ett[i+1].link(s,t);};
            ett[i].edge_update(s,g);
            auto f=[&](int x)->bool{
                for(auto itr=edges[i][x].begin();itr!=edges[i][x].end();){
                    auto y=*itr;
                    itr=edges[i][x].erase(itr);
                    edges[i][y].erase(x);
                    if(edges[i][x].size()==0)ett[i].edge_connected_update(x,0);
                    if(edges[i][y].size()==0)ett[i].edge_connected_update(y,0);
                    if(ett[i].same(x,y)){
                        edges[i+1][x].insert(y);
                        edges[i+1][y].insert(x);
                        if(edges[i+1][x].size()==1)ett[i+1].edge_connected_update(x,1);
                        if(edges[i+1][y].size()==1)ett[i+1].edge_connected_update(y,1);
                    }else{
                        for(int j=0;j<=i;j++){
                            ett[j].link(x,y);
                        }
                        return 1;
                    }
                }
                return 0;
            };
            if(ett[i].try_reconnect(s,f))return 1;
        }
        return 0;
    }
    constexpr static T et=T();
    constexpr static T fn(T s,T t){
        return s+t;
    }
};

signed main() {
  cin.tie(0);
  ios::sync_with_stdio(false);
  int N;
  cin >>N;
  dynamic_connectivity<int>dc(N);
  vvi(V,N-1,3);

  irep(N-1) {
    int A,B,C;
    cin >>A>>B>>C;
    A--;B--;
    dc.link(A,B);
    V[i]={C,A,B};
  }
  Sort(V);
  Reverse(V);
  int ans=0;

  irep(N-1) {
    dc.cut(V[i][1],V[i][2]);
    ans+=V[i][0]*dc.size(V[i][1])*dc.size(V[i][2]);
  }

  cout<<ans<<'\n';
}
}
void TopoSort{//トポロジカルソート（DAG判定にも使用可）
  /* topo_sort(G): グラフG をトポロジカルソート
    返り値: トポロジカルソートされた頂点番号
    計算量: O(|E|+|V|)
    DAG検出：返り値の配列サイズがもとのグラフの頂点数と等しければDAG
 */

  vector<int> topo_sort(const Graph &G) {  // bfs
    vector<int> ans;
    int n = (int)G.size();
    vector<int> ind(n);            // ind[i]: 頂点iに入る辺の数(次数)
    for (int i = 0; i < n; i++) {  // 次数を数えておく
        for (auto e : G[i]) {
            ind[e]++;
        }
    }
    queue<int> que;
    for (int i = 0; i < n; i++) {  // 次数が0の点をキューに入れる
        if (ind[i] == 0) {
            que.push(i);
        }
    }
    while (!que.empty()) {  // 幅優先探索
        int now = que.front();
        ans.push_back(now);
        que.pop();
        for (auto e : G[now]) {
            ind[e]--;
            if (ind[e] == 0) {
                que.push(e);
            }
        }
    }
    return ans;
}

  // 使用例（DAG検出として）https://atcoder.jp/contests/abc216/submissions/25450358
  signed main() {
  int N,M;
  cin >>N >>M;
  Graph G(N);
  irep(M) {
    int K;
    cin >>K;
    int before=-1;
    jrep(K) {
      int A;
      cin >>A;
      A--;
      if (before==-1) {
        before=A;
        continue;
      }
      else {
        G[before].push_back(A);
      }
    }
  }
  vector<int> A=topo_sort(G);
  // cout<<A.size()<<endl;
  if (A.size()==N) cout<<"Yes"<<endl;
  else cout<<"No"<<endl;
}

}

