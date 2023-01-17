#include<bits/stdc++.h>
#include <time.h>
#include <vector>
#include <algorithm>
#define nMax  1000
#define cMax  1000000.0
#define kMax  1000
#define W 1000000
#define NUM_INDIVIDUAL 500
#define maxGen 1000

using namespace std;
double w[nMax][nMax];
int n, K, alpha;
double ans = cMax * nMax * 1.00;

double fit[NUM_INDIVIDUAL];
int Sol[nMax];
int population[NUM_INDIVIDUAL][nMax];
double bestFit = cMax * nMax * 1.00;
double avaiFit = cMax * nMax * 1.00;
int loseStep = 0;
double totalWeight = 0.00;
int KsizeMax = 0;
vector<int> surviver;
vector<int> terminated;

float selec_percent = 0.7;
float mutation_percent = 0.01;

void input(const char* path); 
int random(int minN, int maxN); 
void generate(); 
void selected();
void crossOver();
void mutation(); 
void fitness(); 
double calTotalWeight();
double calObjective(int iInvi);
int violation(int iInvi);
int tournamentSelect();
void getBestFit();
void solve(int _k, int _alpha, const char* path);
pair<vector<vector<int> > , vector<vector<int> > > cross(int iInvi1, int iInvi2);
int main(){
    solve(2, 2, "datas/small/dense_20.txt"); // k , alpha, file input 
	return 0;
}
void solve(int _K, int _alpha, const char* path){
    double start = time(0);
    srand((int)time(0));
	input(path);
    K = _K;
	alpha = _alpha;
    KsizeMax = int((n + K * alpha - alpha)/ K);
    generate();
    totalWeight = calTotalWeight();
    int generation = maxGen;
    while(generation){
    	if(loseStep >= 100){
    		srand((int)time(0));
    		loseStep = 0;
    		bestFit = min(bestFit, avaiFit);
    		avaiFit = cMax;
    		generate();
		}
		fitness();
		selected();
        crossOver(); 
		mutation();
		
		generation--;
	}
	bestFit = min(bestFit, avaiFit);
	cout<<fixed<<setprecision(10)<<bestFit<<endl;
    cout<<loseStep<<endl;
    cout<<time(0) - start<<endl;
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

void generate(){
    for(int i = 0; i < NUM_INDIVIDUAL; i++){
		for(int j = 0; j < n; j++){
			int r = random(0, K - 1);
			population[i][j] = r;
		}
	}
}

int violation(int iInvi){
	int Ksize[K];	
	memset(Ksize, 0, sizeof(Ksize));
	int violation = 0;
	for(int i = 0; i < n; i++){
		Ksize[population[iInvi][i]]++;
	}
	for(int i = 0; i < K - 1; i++){
		for(int j = i + 1; j < K; j++){
			violation += max(abs(Ksize[i] - Ksize[j]) - alpha, 0);
		}
	}
	return violation;
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

double calObjective(int iInvi){
	vector<vector<int> > P (K) ;
	double inCost = 0.0;
	for(int i = 0; i < n; i++){
		P[population[iInvi][i]].push_back(i);
	}
	for(int x = 0; x < K; x++){
		if(!P[x].size()) continue;
		for(int i = 0; i < P[x].size() - 1; i++){
			for(int j = i + 1; j < P[x].size(); j++){
				inCost += w[P[x][i]][P[x][j]];
			}
		}
	}
	return totalWeight - inCost;
}

void fitness(){
	for(int  i = 0; i < NUM_INDIVIDUAL; i++){
		fit[i] = calObjective(i) + W * violation(i);
	}
	if(fit[0] < avaiFit){
		avaiFit = fit[0];
		loseStep = 0;
	}
	else{
		loseStep++;
	}
}

void mutation(){
	for(int i = 0; i < int(mutation_percent * NUM_INDIVIDUAL); i++){
	    int mutatePoint = random(0, NUM_INDIVIDUAL - 1);
	    int mutatePosition = random(0, n - 1);
	    int mutateValue = random(0, K - 1);
	    population[mutatePoint][mutatePosition] = mutateValue;
    }
}

void selected(){
	double fit_tmp[NUM_INDIVIDUAL];
	for(int i = 0; i < NUM_INDIVIDUAL; i++){
		fit_tmp[i] = fit[i];
	}
	sort(fit_tmp, fit_tmp + NUM_INDIVIDUAL);
	double THRESH_HOLD = fit_tmp[int(NUM_INDIVIDUAL * selec_percent)];
	surviver.clear();
	terminated.clear();
    for(int i = 0; i < NUM_INDIVIDUAL; i++){
    	if(fit[i] > THRESH_HOLD){
    		terminated.push_back(i);
		}
		else surviver.push_back(i);
    }
}

int tournamentSelect(){
	int first = random(0, surviver.size() - 1);
	int second = first;
	while(second == first){
		second = random(0, surviver.size() - 1);
	}
	return fit[surviver[first]] >= fit[surviver[second]] ? first : second;
}

pair<vector<vector<int> > , vector<vector<int> > > cross(int iInvi1, int iInvi2){
    vector<vector<int> > first (K) ;
	vector<vector<int> > second (K) ;
	pair<vector<vector<int> >, vector<vector<int> > > childs;
	
	int child1[n];
	int child2[n];

	int point = random(0, n - 1);
	for(int  i = 0; i < point; i++){
		child1[i] = population[iInvi1][i];
		child2[i] = population[iInvi2][i];
	}
	for(int  i = point; i < n; i++){
		child1[i] = population[iInvi2][i];
		child2[i] = population[iInvi1][i];
	}
	for(int i = 0; i < n; i++){
		first[child1[i]].push_back(i);
		second[child2[i]].push_back(i);
	}
	childs.first = first;
	childs.second = second;
	
	return childs;
}

void crossOver(){
	int numTerminated = terminated.size();
	int numCross = ceil(numTerminated/ 2);
	for(int i = 0; i < numCross; i++){
        int first = tournamentSelect();
        int second = tournamentSelect();
        pair<vector<vector<int> > , vector<vector<int> > > childs = cross(first, second);
        int addPosition = terminated.back();
        for(int i = 0; i < K; i++){
        	for(int j = 0; j < childs.first[i].size(); j++){
        		population[addPosition][childs.first[i][j]] = i;
			}
		}
		terminated.pop_back();
		if(!terminated.empty()){
		    addPosition = terminated.back();
		    for(int i = 0; i < K; i++){
        	    for(int j = 0; j < childs.second[i].size(); j++){
        		    population[addPosition][childs.second[i][j]] = i;
			    }
		    }	
		    terminated.pop_back();
		}
	}
}