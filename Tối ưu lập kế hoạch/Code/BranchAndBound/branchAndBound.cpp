#include<bits/stdc++.h>
#include <time.h>
#include <vector>
#define nMax  1000
#define cMax  1000000
#define kMax  1000

using namespace std;


double w[nMax][nMax];
int n, K, alpha;
int P[nMax]; 
int Ksize[kMax];
int KsizeMax = 0;
double totalWeight = 0.00;
double ans = cMax * nMax * 1.00; // get better value from Greedy 

void input(const char* path); 
int random(int minN, int maxN);
double calTotalWeight(); 
void Try(int idx, double cutSize);
void solution(double cutSize);
void solve(int _K, int _alpha, const char* path);
int main(){
    solve(2, 2, "datas/small/dense_20.txt"); // K, alpha , input file

	return 0;
}

void solve(int _K, int _alpha, const char* path){
	input(path);
    K = _K;
    alpha = _alpha;
    KsizeMax = int((n + K * alpha - alpha)/ K);
    totalWeight = calTotalWeight();
    memset(P, -1, sizeof(P));
    memset(Ksize, 0, sizeof(Ksize));
    double start = time(0);
    Try(0, 0.00);
    cout<<fixed<<setprecision(10)<<"weigth: "<<ans<<endl;
    cout<<"Time elapsed: "<<(time(0) - start) * 1000;
}

void Try(int idx, double cutSize){
	if(idx == n){
		solution(cutSize);
		return;
	}
    double eachPartionCutSize[K];
    memset(eachPartionCutSize, 0.00, sizeof(eachPartionCutSize));
    double totalCut = 0.0;
    for(int i = 0; i < n; i++){
        if(P[i] != -1){
            eachPartionCutSize[P[i]] += w[idx][i];
            totalCut += w[idx][i];
        }
    }
      
	for(int i = 0; i < K; i++){
        if(Ksize[i] + 1 > KsizeMax) continue;
        if(cutSize + totalCut - eachPartionCutSize[i] > ans) continue;

        P[idx] = i;
        Ksize[i]++;
        Try(idx + 1, cutSize + totalCut - eachPartionCutSize[i]);
        Ksize[i]--;
        P[idx] = -1;

	}
}

void solution(double cutSize){ 
    int KsizeMin = nMax;
    int KsizeMax = 0;

    for(int i = 0; i < K; i++){
        KsizeMax = max(Ksize[i], KsizeMax);
        KsizeMin = min(Ksize[i], KsizeMin);
    }
    if(KsizeMax - KsizeMin > alpha || KsizeMin == 0) return;
    ans = min(cutSize, ans);
}

double calTotalWeight(){
	double total = 0.0;
	for(int i = 0; i < n - 1; i++){
		for(int j = i + 1; j < n; j++){
			total += w[i][j];
		}
	}
	return total;
}

int random(int minN, int maxN){
    return minN + rand() % (maxN + 1 - minN);
}

void input(const char* path){
    int m = 0;
	freopen(path, "r", stdin);
	cin>>n>>m;
	for(int i = 0; i < m; i++){
	    int u, v;
		double c;
		cin>>u>>v>>c;
		w[u][v] = c;
		w[v][u] = c;	
	}
	fclose(stdin);
}