#include <bits/stdc++.h>
using namespace std;

void time() { //時間を計測して使用
  // #define ENDLOOP clock_t END = clock(); if(((long double)(END - START) / CLOCKS_PER_SEC * 1000.0) > TIME) break;
  // constexpr long double TIME = 2.0* 98000.0 / 100.0;
  // int main() {
  //   clock_t START;
  //   ENDLOOP
  // }
}
int iota() { //iota
  iota(V.begin(),V.end(),K); //K,K+1,K+2...Nの配列にする。
}
int vector() { //vector
  find(V.begin(),V.end(),3)-V.begin() //3のある位置（0オリジン）を求める
  int (){ //erase
  vector<int> V={1,2,3,4};
  cout<<V[2]<<endl;  //3
  V.erase(V.begin()+2);
  cout<<V[2]; //4
  }
  V.clear(); //全要素削除
}
int string() {//string
// メンバ関数find
if (S.find(T)+1) //文字列のSの部分文字列に文字列Tが存在するか


// メンバ関数compare
//文字列Sのs文字目~s+l文字目までがabcと一致するか否か
//一致していれば0が、不一致であればその演算結果が帰ってくる
// https://marycore.jp/prog/cpp/std-string-equal-compare/
S.compare(s,l,"abc")
//例
std::string s = "a-b-c";
s.compare(1, 3, "-b-"); // 0
s.compare(3, 2, "-c");  // 0

s.compare(0, 2, "a-b");    // -1
s.compare(0, 2, "a-b", 2); //  0
s.compare(0, 2, std::string("-a-"), 1, 2); // 0
}
int find() { //文字列の存在判定
 int main(){
   string S="ABCDE";
   int A=S.find("CDE");
   cout<< A<<endl; //2
 }
}
int cast(){  //文字列⇔数値変換
  #include <bits/stdc++.h>
using namespace std;

int main() {
  cout << (int)'A' << endl;
  cout << (int)'B' << endl;
  cout << (int)'Z' << endl;

  cout << (char)65 << endl;
  cout << (char)66 << endl;
  cout << (char)90 << endl;
}

//文字としての数値を数値に変換する(ASCIIの順番を利用)
int  one ='1'-'0' //=1;

//数値列を文字列に変換
to_string(num);
//文字列を数値列に変換
stoi("1010");

// K進数の文字列を10進数に変換
signed main() {
  int K;
  string A,B; 
  cin >>K >>A>>B;
  long C=stol(A,nullptr,K);
  long D=stol(B,nullptr,K);
  cout<<C*D<<endl;
}

}
int substr(){ //文字列の一部を取得
宣言
std::string str = 文字列;
str.substr(開始位置, 取り出す長さ); //開始位置から取り出す長さ分の文字列を取得
str.substr(開始位置) //開始位置から最後までの文字列を取得
例
std::string str = "samurai,engineer,se!";
std::string substr = str.substr(8, 8);
std::cout << substr << std::endl;
}
int Hairetsu(){ //配列
vector<int> vec(N);
for (int i = 0; i < N; i++) {
    cin >> vec.at(i);
}
V.back() //最後の文字
V.front() //最初の文字
vec.push_back(10); // 末尾に10を追加
}
int n^Hairetu(){ //二次元配列
  vector<vector<要素の型>> 変数名(縦の要素数, vector<要素の型>(横の要素数));
  vector<vector<int>> data(3, vector<int>(4));　//3行4列の配列
}
int pair_tuple() { //pair
//宣言
pair<型1, 型2> 変数名;
pair<型1, 型2> 変数名(値1, 値2);
//example
pair<pair<string,int>,int> p[110];
//アクセス
変数名.first   // 1つ目の値
変数名.second  // 2つ目の値
//生成
make_pair(値1, 値2)
//宣言（tuple）//n個の組を表現
tuple<型1, 型2, 型3, ...(必要な分だけ型を書く)> 変数名;
tuple<型1, 型2, 型3, ...(必要な分だけ型を書く)> 変数名(値1, 値2, 値3, ...); // 初期化
//pair/tupleの比較
型が同じpairやtuple同士は比較することができます。
例えばpair<int, int>を比較する場合、1番目の値を基準に比較され、もし1番目の値が等しい場合は2番目の値を基準に比較されます。

//使用例
int example() {
  #include <bits/stdc++.h>
using namespace std;
#define ll long long
#define irep(n) for (int i=0; i < (n); ++i)
#define irepf1(n) for (int i=1; i <= (n); ++i)
#define jrep(n) for (int j=0; j < (n); ++j)
#define jrepf1(n) for (int j=1; j <= (n); ++j)
#define PI 3.14159265358979323846264338327950288
//fixed << setprecision(10) <<

int main(){
  int N;
  cin >>N;
  pair<pair<string,int>,int> p[110];
  irep(N) {
    string S;
    int P;
    cin >>S >>P;
    p[i]=make_pair(make_pair(S,-P),i+1);
  }
  sort(p,p+N);
  irep(N) {
    cout<<p[i].second<<endl;
  } 
}
}
}
int function() { //関数
返り値の型 関数名(引数1の型 引数1の名前, 引数2の型 引数2の名前, ...) {処理}
}
int HanniFor() { //範囲for文
  vector<int> a = {1, 3, 2, 5};
  for (int x : a) {
    cout << x << endl;
  }
}
int tuple() { //tuple
vector<tuple<int, int, int>> a;
}
int eiriasu() { //型エイリアス
using vi = vector<int>; // intの1次元の型に vi という別名をつける
using vvi = vector<vi>; // intの2次元の型に vvi という別名をつける
}
int map(){ //map
     map<Keyの型, Valueの型> 変数名;
追加:変数[key] = value;
削除:変数.erase(key);
アクセス:変数.at(key)
所属判定:変数.count(key)
要素数の取得;変数.size()
例:
・宣言
  map<string, int> score;  // 名前→成績
  score["Alice"] = 100;
  score["Bob"] = 89;
  score["Charlie"] = 95;

  cout << score.at("Alice") << endl;   // Aliceの成績
  cout << score.at("Bob") << endl;     // Bobの成績
  cout << score.at("Charlie") << endl; // Daveの成績
・判定
if (score.count("Alice")) {
  cout << "Alice: " << score.at("Alice") << endl;
}
if (score.count("Jiro")) {
  cout << "Jiro: " << score.at("Jiro") << endl;
}
・ループ
// Keyの値が小さい順にループ
for (auto p : 変数名) {
  auto key = p.first;
  auto value = p.second;
  // key, valueを使う
}
for (auto p : score) {
  auto k = p.first;
  auto v = p.second;
  cout << k << " => " << v << endl;
}


 }
int queue() { //キュー
    #include <bits/stdc++.h>
using namespace std;

int main() {
  queue<int> q;
  q.push(10);
  q.push(3);
  q.push(6);
  q.push(1);

  // 空でない間繰り返す
  while (!q.empty()) {
    cout << q.front() << endl;  // 先頭の値を出力
    q.pop();  // 先頭の値を削除
  }
}

}
int priority queue() { //優先度付きキュー
int main() {
  priority_queue<int> pq;
  pq.push(10);
  pq.push(3);
  pq.push(6);
  pq.push(1);

  // 空でない間繰り返す
  while (!pq.empty()) {
    cout << pq.top() << endl;  // 最大の値を出力
    pq.pop();  // 最大の値を削除
  }
}
・最小の要素を取り出す
int main() {
  // 小さい順に取り出される優先度付きキュー
  priority_queue<int, vector<int>, greater<int>> pq;
  
  priority_queue<int> pq;
  pq.push(10);
  pq.push(3);
  pq.push(6);
  pq.push(1);

  // 空でない間繰り返す
  while (!pq.empty()) {
    cout << pq.top() << endl;  // 最小の値を出力
    pq.pop();  // 最小の値を削除
  }
}
}
int set() { //セット
使用例
変数.erase(値);
  変数.size()
  変数.empty()  // 空ならtrueを返す
*begin(変数)　//最小値の取得
*rbegin(変数) //最大値の取得
auto it=S.lower_bound(A) //二分探索のイテレータ
int A=*S.lower_bound(A) //二分探索の値
for (auto value : 変数名) { //ループ
  // valueを使う
  cout <<value<<endl;
}

int main() {
  set<int> S;
  S.insert(3);
  S.insert(7);
  S.insert(8);
  S.insert(10);
  // 既に3は含まれているのでこの操作は無視される
  S.insert(3);


  // 集合の要素数を出力
  cout << "size: " << S.size() << endl;

  // 7が含まれるか判定
  if (S.count(7)) {
    cout << "found 7" << endl;
  }
  // 5が含まれるか判定
  if (S.count(5)) {
    cout << "found 5" << endl;
  }
}
}
int multiset() { //マルチセット
    //変数宣言（空集合）
  multiset<int> s;
  //要素の挿入
  s.insert(1),s.insert(3),s.insert(3),s.insert(4),s.insert(4),s.insert(5);
  //各要素へのアクセス
  for(auto i=s.begin();i!=s.end();i++)cout << *i << endl;//1 3 3 4 4 5

  //要素数の確認
  auto c = s.count(3);
  cout << c << endl; //2

  //要素の削除（重複している場合はすべて消す）
  c = s.erase(3);
  cout << c << endl; //2
  for(auto i=s.begin();i!=s.end();i++)cout << *i << endl;//1 4 4 5

  //要素の削除（重複している場合は1つだけ消す）
  //集合にない要素を削除しようとするとエラーとなるため、eraseの前にfindしておく。
  auto it = s.find(4);
  if (it != s.end()) s.erase(it);
  for(auto i=s.begin();i!=s.end();i++)cout << *i << endl;//1 4 5
}
int PBDS() { //Policy Based Data Structure
//参考 https://xuzijian629.hatenablog.com/entry/2018/12/01/000010

// tree
// 宣言（intの部分をpair<int,int>などにもできる）
// 基本的にはSetと同じ使い方
tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> S;

// 使用方法
S.insert(A);
S.erase(A);
S.find_by_order(A) //A番目の要素のイテレータを返す
S.order_by_key(A) //A未満の要素の数を返す

// hash table
// 宣言
// 使い方はunordered_mapと同様
gp_hash_table<int, int> m;
}
int stack() { //スタック
    #include <bits/stdc++.h>
using namespace std;

int main() {
  stack<int> s;
  s.push(10);
  s.push(1);
  s.push(3);

  cout << s.top() << endl;  // 3 (最後に追加した値)
  s.pop();
  cout << s.top() << endl;  // 1 (その前に追加した値)
}

}
int deque() { //デック
    #include <bits/stdc++.h>
using namespace std;

int main() {
  deque<int> d;
  d.push_back(10);
  d.push_back(1);
  d.push_back(3);

  // この時点で d の内容は { 10, 1, 3 } となっている

  cout << d.front() << endl; // 10 (先頭の要素)
  d.pop_front();  // 先頭を削除

  // この時点で d の内容は { 1, 3 } となっている

  cout << d.back() << endl;  // 3 (末尾の要素)
  d.pop_back();  // 末尾を削除

  // この時点で d の内容は { 1 } となっている

  d.push_front(5);
  d.push_back(2);

  // この時点で d の内容は { 5, 1, 2 } となっている

  cout << d.at(1) << endl; // 1
}

}
int struct() { //構造体
  ・宣言方法
  struct 構造体名 {
  型1 メンバ変数名1
  型2 メンバ変数名2
  型3 メンバ変数名3
  ...(必要な分だけ書く)
};  // ← セミコロンが必要
  ・例
  #include <bits/stdc++.h>
using namespace std;

struct MyPair {
  int x;     // 1つ目のデータはint型であり、xという名前でアクセスできる
  string y;  // 2つ目のデータはstring型であり、yという名前でアクセスできる
};

int main() {
  MyPair p = {12345, "hello"};  // MyPair型の値を宣言
  cout << "p.x = " << p.x << endl;
  cout << "p.y = " << p.y << endl;
}

}
int bitset() { //ビット演算
  ・宣言
  bitset<ビット数> 変数名;  // すべてのビットが0の状態で初期化される
  bitset<ビット数> 変数名("ビット列(長さはビット数に合わせる)");  // 指定したビット列で初期化される
  （例）bitset<4> b("1010");
  ・演算子：AND& OR| XOR^ NOT~ 左シフト<< 右シフト>> 
  b.to_ulong() //unsinged_longへ変換
  b.to_ullong() //unsinged_long longへ変換
  b.to_string() //stirngへ変換
  b.count() //立っているbitの数を数える。
  // 注意！！bit演算の結果をintに入れようとすると良くないことが起こる！！
  // 例: int A= BITSET.test(i)
  // こういうときはbitset<1> Aと宣言してからBITSET.test(i)を代入する
  ・サンプルプログラム
  #include <bits/stdc++.h>
using namespace std;

int main() {
  bitset<8> a("00011011");
  bitset<8> b("00110101");

  auto c = a & b;
  cout << "1: " << c << endl;         // 1: 00010001
  cout << "2: " << (c << 1) << endl;  // 2: 00100010
  cout << "3: " << (c << 2) << endl;  // 3: 01000100
  cout << "4: " << (c << 3) << endl;  // 4: 10001000
  cout << "5: " << (c << 4) << endl;  // 5: 00010000

  c <<= 4;
  c ^= bitset<8>("11010000"); // XOR演算の複合代入演算子
  cout << "6: " << c << endl; // 6: 11000000
}

#include <bits/stdc++.h>
using namespace std;
  int main() {
    bitset<4> S;
    S.set(0, 1);  // 0番目のビットを1にする
    cout << S << endl;

    if (S.test(3)) {
      cout << "4th bit is 1" << endl;
    } else {
      cout << "4th bit is 0" << endl;
    }
  }

  int main() { //ビット全探索
    // 3ビットのビット列をすべて列挙する
    for (int tmp = 0; tmp < (1 << 3); tmp++) { //1 << 3 は1のビット左シフト演算（1000=2^3=8）,2^kを得たいときに使用する)
      bitset<3> s(tmp);
      // ビット列を出力
      cout << s << endl;
    }
  }

  int main() { //・ビット全探索のひな形
  for (int tmp = 0; tmp < (1 << ビット数); tmp++) {
  bitset<ビット数> s(tmp);
  // (ビット列sに対する処理)
}
  }
  int example() { //ビット全探索を使った問題の例

int main () {
  int N, K;
  cin >> N >> K;
  vector<int> A(N);
  for (int i = 0; i < N; i++) {
    cin >> A.at(i);
  }

  bool ans = false;

  // すべての選び方を試して、総和がKになるものがあるかを調べる
  for (int tmp = 0; tmp < (1 << 20); tmp++) {
    bitset<20> s(tmp);  // 最大20個なので20ビットのビット列として扱う

    // ビット列の1のビットに対応する整数を選んだとみなして総和を求める
    int sum = 0;
    for (int i = 0; i < N; i++) {
      if (s.test(i)) {
        sum += A.at(i);
      }
    }
    if (sum == K) {
      ans = true;
    }
  }

  if (ans) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}

  }
  int example2() { //ABC167C https://atcoder.jp/contests/abc167/tasks/abc167_c
  int main(){
  int N,M,X;
  cin >>N>>M>>X;
  vector<int> C(N);
  vector<int> D(M);
  vector<vector<int>> V(N,vector<int>(M)); 
  irep (N) {
    cin >>C[i];
    jrep (M) cin >>V[i][j];
  }
  ll ans=10000000;
  for (int bit=0;bit < (1<<N); bit++) {
    ll Price=0;
    irep (M) D.at(i)=0;
    for (int i=0; i<N;i++) {
      if (bit & (1<<i)) {
        Price+=C.at(i);
        for (int j=0;j<M;j++) D.at(j)+=V.at(i).at(j);
      }
    }
    bool flag=true;
    irep (M) if (D.at(i)<X) flag=false;
    if (flag) ans=min(ans,Price);
  }
  if (ans!=10000000) cout<<ans<<endl;
  else cout<<-1<<endl;
}
  }
}
int sort() { //ソート (文字列でも可)
sort(vec.begin(), vec.end()) //昇順
reverse(vec.begin(), vec.end()) //逆にする
}
int permutation() { //順列　（全探索にも応用可）
使用例
int main() {
  vector<int> Vec  {1,2,3,4}; //配列の宣言
  do{
      for(int i=0; i<4;i++){ //そのときの配列の全要素を出力
    	cout << Vec.at(i);
        if (i!=3) cout <<" ";
      }
      cout<<endl;
  }while(next_permutation(Vec.begin(), Vec.end())); //（辞書順で考えて）次の順列が存在すれば継続
}
二次元配列などの場合は行の長さを表す変数indexを使用するといい
int main (){ //N個の街を周回する問題　（https://atcoder.jp/contests/abc183/tasks/abc183_c）
	int n,k;
	cin >> n >> k;
	vector<vector<int>>T(n,vector<int>(n));
	for(int i=0;i<n;i++)for(int j=0;j<n;j++){
		cin >> T[i][j];
	}
	
	vector<int>index;
	for(int i=0;i<n;i++)index.push_back(i);
 
	int ans=0;
	do{
		int time=0;
		for(int i=0;i<n;i++)time+=T[index[i]][index[(i+1)%n]];
		if(time==k)ans++;
	}while(next_permutation(index.begin()+1, index.end()));
	cout << ans;
}
}
int bound() { //昇順ソートされた配列でn以上/以下の最小の要素を取得
  #include <bits/stdc++.h>
using namespace std;

int main() {
  vector<int> a = {0, 10, 13, 14, 20};
  // aにおいて、12 以上最小の要素は 13
  cout << *lower_bound(a.begin(), a.end(), 12) << endl; // 13

  // 14 以上最小の要素は 14
  cout << *lower_bound(a.begin(), a.end(), 14) << endl; // 14

  // 10 を超える最小の要素は 13
  cout << *upper_bound(a.begin(), a.end(), 10) << endl; // 13

  //求めた値の「位置」を求める
  auto it=lower_bound(a.begin(),a.end(),10); //イテレータ
  int id=it-a.begin() //イテレータから「位置」を求める
  cout<<id<<endl; //1
  cout<<a[id]>>endl; //10

}
 //
}
int swap() { //値の交換
  swap(a, b)//a,bの値を交換する
}
int gcd() { //最大公約数
  gcd(a,b) //aとbの最大公約数を求める
  a/gcd(a, b) * b //aとbの最小公倍数を求めることができる
}
int count(x,l,r) { //配列の中でxがいくつ存在するかを調べる (r-l)
count(a + l, a + r, x)

使用例
int main() {
    // 例 1: 配列 a に含まれる 1 の個数、2 の個数を出力する（それぞれ　4, 3 と出力されます）
    int a[10] = {1, 2, 3, 4, 1, 2, 3, 1, 2, 1};
    cout << count(a, a + 10, 1) << endl;
    cout << count(a, a + 10, 2) << endl;

    // 例 2: b[1], b[2], ..., b[N] を受け取り、その後 Q 個の質問を受け取る。
    // 各質問に対し、b[l], b[l+1], ..., b[r] の中で x が何個あるかを出力する。
    int b[1009], N, Q;
    cin >> N;
    for (int i = 1; i <= N; i++) cin >> b[i];
    cin >> Q;
    for (int i = 1; i <= Q; i++) {
        int l, r, x;
        cin >> l >> r >> x;
        cout << count(b + l, b + r + 1, x) << endl;
    }
    return 0;
}
{string S;
cin >>S;
cout<<count(S.begin(),S.end(),'a')<<endl;
}
}
int modInverse {//逆元 × mod
  // A/B(mod P)をしたい時はBの逆元P^-1(mod P)を求め、A*B^-1で求められる
}


