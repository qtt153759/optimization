#include<bits/stdc++.h>
#include<time.h>
#include<vector>
#include<algorithm>
#define nMax  1000
#define cMax  1000000.0
#define kMax  1000

using namespace std;

double w[nMax][nMax];
int n, K, alpha;
double ans = cMax * nMax * nMax * 1.00;
int P[nMax];
vector<int> V;

void input(const char* path); 
int random(int minN, int maxN); 
double totalGain(int idx, int par);
double internalGain(int idx, int par);
void constructive(int _k, int _alpha, const char* path);
int chooseVertex(int par);

int main(){
    memset(P, -1, sizeof(P));
    constructive(6, 10, "datas/huge/dense_400.txt"); // k , alpha, file input 
	return 0;
}

void constructive(int _K, int _alpha, const char* path){
    double start = time(0);
    double cutSize = 0.00;
    srand((int)time(0));
	input(path);
    K = _K;
	alpha = _alpha;
    int cur_par = 0;

    for(int i = 0; i < n; i++){
        V.push_back(i); 
    }
    // khoi tao K seed cho moi partition
    for(int i = 0; i < K; i++){
        int r = random(0, V.size() - 1);
        P[V.at(r)] = i;
        cutSize += totalGain(V.at(r), i);
        cur_par++;
        V.erase(V.begin() + r);
    }

    // gan cacs dinh con lai theo vong tron
    while(!V.empty()){
        cur_par = cur_par % K;
        int chosed = chooseVertex(cur_par);
        P[V.at(chosed)] = cur_par;
        cutSize += totalGain(V.at(chosed), cur_par);
        cur_par++;
        V.erase(V.begin() + chosed);
    }
    // for(int i = 0; i < n; i++){
    //     cout<<P[i]<<" ";
    // }
    cout<<fixed<<setprecision(10)<<"weigth: "<<cutSize<<endl;
    cout<<"Time elapsed: "<<(time(0) - start) * 1000;

}

int chooseVertex(int par){
    int objective[V.size()];
    vector<int> bestChoice;
    int bestObj = nMax * cMax;
    for(int i = 0; i < V.size(); i++){
        objective[i] = totalGain(V.at(i), par) - internalGain(V.at(i), par);  
        bestObj = min(bestObj, objective[i]);    
    }
    for(int i = 0; i < V.size(); i++){
        if(objective[i] == bestObj)
            bestChoice.push_back(i);
    }
    int r = random(0, bestChoice.size() - 1);
    return bestChoice.at(r);
}

double totalGain(int idx, int par){
    double gain = 0.00;
    for(int i = 0; i < n; i++){
        gain += w[idx][i] * ((P[i] != -1 && P[i] != par) ? 1 : 0);
    }
    return gain;
}

double internalGain(int idx, int par){
    double gain = 0.00;
    for(int i = 0; i < n; i++){
        gain += w[idx][i] * (P[i] == par ? 1 : 0);
    }
    return gain;
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



