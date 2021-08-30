int times = (A + B - 1) / B; //切り上げ(A/B)
cout << fixed << setprecision(10)　<< double << endl; //小数点以下桁数指定
a() {long long int a = 999999999999999999; //10^18 - 1
long long int b = a;
a += a / 100; //=> 1009999999999999998(正しい)
b *= 1.01; //=> 1010000000000000000(間違ってる)
}
int example() { //カレンダー的文字列の比較なら辞書式の比較が有効
if (s <= "2019/04/30") cout << "Heisei" <<endl;
}
#define rep(i,n) for (int i=0; i < (n); ++i) //ヘッダーで宣言しておくと本文でrep(i,n) cin >>vec.at(i)のように使用できる。
const double PI=acos(-1.0); //円周率
S = S.back() + S.substr(0,S.size()-1); //文字列の回転
lcm(A,B) //AとBの最小公倍数を求める
int  () {// a 以上 b 以下の整数のうち条件を満たすものの個数を求める問題です．このような問題では，
// f(n) := 0 以上 n 以下の整数のうち条件を満たすものの個数
// と定義しておくと，答えは f(b) − f(a − 1) で求まるので楽です．ただし，a = 0 のときに f(−1) が呼ばれる
// ことに注意してください．このことに注意すると，f は次のように書けます．
}
for (auto value:SS) { //set SSの中にMMの要素たり得る要素を全て保存しておいて、MMの要素判定に使用 
        if (value!=C) MM.at(value)=0;
        else {
          MM.at(value)=Size-i;
        }
      }
() {// ↑（追記）　auto変数の宣言時に変数名前に&をつけることで対応可能
例
int main() {
  map<int,int> M;
  M[1] =2;
  M[2]=5;
  for (auto x:M) cout<<x.second<<endl;
  for (auto& x:M) x.second=0;
  for (auto x:M) cout<<x.second<<endl;
}}
//マンハッタン距離について
// https://img.atcoder.jp/abc178/editorial-E-phcdiydzyqa.pdf
// 2つの閉区間[a,b],[c,d]が共通部分を持つかの判定は,max(a,c)<=min(b,d)で可能
// https://atcoder.jp/contests/abc207/editorial/2152
ans += (max(l[i],l[j]) <= min(r[i],r[j]))

// 互いに素な2つの自然数a,bの組み合わせで表現不可能な最大の数はmn-m-nである（Chicken McNugget Theorem）
// https://codeforces.com/blog/entry/91195

・確認事項
// TLEしたら制約に注目！！！
// 組み合わせを求めるときに、factorial(nCk)を使用すると値が大きくなりすぎてオーバーフローする可能性がある。可能であれば計算はシンプルに（例:nC2であればn(n-1)/2で求められる）
// 無限ループしている時はループのカウンタがオーバーフローしていないかチェック（int i=0;~~でiが10^6以上になったりしていないか）
//ループの回数が分からない場合、どんなループになるかを簡単に確認すると良い
//mapの値はfor (auto)の中で書き換えることは出来ない。書き換えたい場合は個々に書き換える（以下例）
//浮動小数点演算はできるだけ避ける（EX:三平方の定理。精度の問題による。）
// 浮動小数点演算を避ける手段として、除算を使用することが挙げられる

