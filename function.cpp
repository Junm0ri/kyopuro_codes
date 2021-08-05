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
    set(int i, T x), build(): i番目の要素をxにセット。まとめてセグ木を構築する。O(n)
    update(i,x): i 番目の要素を x に更新。O(log(n))
    query(a,b): [a,b) での最小の要素を取得。O(log(n))
    find_rightest(a,b,x): [a,b) で x以下の要素を持つ最右位置を求める。O(log(n))
    find_leftest(a,b,x): [a,b) で x以下の要素を持つ最左位置を求める。O(log(n))
*/
template <typename T>
struct RMQ {
    const T e = numeric_limits<T>::max();
    function<T(T, T)> fx = [](T x1, T x2) -> T { return min(x1, x2); };
    int n;
    vector<T> dat;
    RMQ(int n_) : n(), dat(n_ * 4, e) {
        int x = 1;
        while (n_ > x) {
            x *= 2;
        }
        n = x;
    }

    void set(int i, T x) { dat[i + n - 1] = x; }
    void build() {
        for (int k = n - 2; k >= 0; k--) dat[k] = fx(dat[2 * k + 1], dat[2 * k + 2]);
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

    // the minimum element of [a,b)
    T query(int a, int b) { return query_sub(a, b, 0, 0, n); }
    T query_sub(int a, int b, int k, int l, int r) {
        if (r <= a || b <= l) {
            return e;
        } else if (a <= l && r <= b) {
            return dat[k];
        } else {
            T vl = query_sub(a, b, k * 2 + 1, l, (l + r) / 2);
            T vr = query_sub(a, b, k * 2 + 2, (l + r) / 2, r);
            return fx(vl, vr);
        }
    }

    int find_rightest(int a, int b, T x) { return find_rightest_sub(a, b, x, 0, 0, n); }
    int find_leftest(int a, int b, T x) { return find_leftest_sub(a, b, x, 0, 0, n); }
    int find_rightest_sub(int a, int b, T x, int k, int l, int r) {
        if (dat[k] > x || r <= a || b <= l) {  // 自分の値がxより大きい or [a,b)が[l,r)の範囲外ならreturn a-1
            return a - 1;
        } else if (k >= n - 1) {  // 自分が葉ならその位置をreturn
            return (k - (n - 1));
        } else {
            int vr = find_rightest_sub(a, b, x, 2 * k + 2, (l + r) / 2, r);
            if (vr != a - 1) {  // 右の部分木を見て a-1 以外ならreturn
                return vr;
            } else {  // 左の部分木を見て値をreturn
                return find_rightest_sub(a, b, x, 2 * k + 1, l, (l + r) / 2);
            }
        }
    }
    int find_leftest_sub(int a, int b, T x, int k, int l, int r) {
        if (dat[k] > x || r <= a || b <= l) {  // 自分の値がxより大きい or [a,b)が[l,r)の範囲外ならreturn b
            return b;
        } else if (k >= n - 1) {  // 自分が葉ならその位置をreturn
            return (k - (n - 1));
        } else {
            int vl = find_leftest_sub(a, b, x, 2 * k + 1, l, (l + r) / 2);
            if (vl != b) {  // 左の部分木を見て b 以外ならreturn
                return vl;
            } else {  // 右の部分木を見て値をreturn
                return find_leftest_sub(a, b, x, 2 * k + 2, (l + r) / 2, r);
            }
        }
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
    rep(i,n) par[i] = i;
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

    rep(ei, es.size()) {
      Edge& e = es[ei];
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
  rep(i, m) {
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

  //使用例（典型90 21） https://atcoder.jp/contests/typical90/tasks/typical90_u

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
