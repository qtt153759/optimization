#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>


/* VARIABLES INITIALIZATIONS */

int N; // ∑ subjects
int M;	// ∑ rooms
int K;	// ∑ pair (i, j) if i and j are conflicted
std::vector<int> c;	// the capacity of each room
std::vector<int> d;	// the number of students enroll in each subject
std::vector< std::vector<int> > w;	// the 2-dim array represents the conflict between 2 subjects with value 1

int k; // the number of timeslots => optimize
std::vector< int > p; // assign subject i to slot p[i], p[subject] = slot
std::vector< std::vector<int> > m; // m[slot][room] = subject

/* DEFINE FUNCTIONS */

void input();
void TRY(int u, int kip); // backtracking
bool check(int u, int kip); // if subject "u" can be done in slot "kip"
void printSolution();

int solution_count = 0;

void input() {
	// std::string line;
	// std::ifstream inputfile (s); // this needs std::string s as function argument
	// if (inputfile.is_open()) {
	// 	while ( getline(inputfile, line) ) {
	// 		std::cout << line << "\n";
	// 	}
	// 	inputfile.close();
	// }
	// else {
	// 	std::cout << "Unable to open file!\n";
	// }

	std::cin >> N;
	d.resize(N+1, 0);
	w.resize(N+1, std::vector<int> (N+1, 0));
	for (int i = 1; i <= N; i++) {
		std::cin >> d[i];
	}
	std::cin >> M;
	c.resize(M+1, 0);
	for (int i = 1; i <= M; i++) {
		std::cin >> c[i];
	}
	std::cin >> K;
	for (int i = 1; i <= K; i++) {
		int a, b;
		std::cin >> a >> b;
		w[b][a] = 1;
		w[a][b] = 1;
	}
}

bool check(int u, int kip) {
	if (p[u] > 0) {
		return false;
	}
	for (int i = 1; i < w[u].size(); i++) {
		// std::cout << u << ", " << i << "\n";
		if (w[u][i] && p[i] == kip){
			return false;
		}
	}
	return true;
}

void printSolution(std::vector< std::vector<int> > _m) {
	for (int i = 1; i <= k; i++) {
		for (int j = 1; j <= M; j++) {
			std::cout << _m[i][j] << " ";
		}
		std::cout << "\n";
	}
	solution_count++;
	std::cout << "======================\n";
}

/*
** Traversing from slot 1 (var kip), then next is room
** u is the number of scheduled subjects
*/
void TRY(int u, int kip) {
	// if (u == 6) {
	// 	std::cout << "xh\n";
	// }
	if (u == N+1) {
		k = std::min(k, kip);
		printSolution(m);
		return;
	}
	if (kip > k) {
		return;
	}
	for (int r = 1; r <= M; r++) {
		if (m[kip][r] == 0) {
			for (int i = 1; i <= N; i++) {
				// std::cout << r << ", " << i << ", " << u << ", " << kip << "\n";
				// std::cout << d[i] << ", " << c[r] << "\n";
				if (check(i, kip) && d[i] <= c[r]) { 
					m[kip][r] = i;
					// std::cout << m[kip][r] << "\n";
					p[i] = kip;
					// std::cout << p[i] << "\n";
					TRY(u+1, kip);
					m[kip][r] = 0;
					p[i] = -1;
				}
			}
		}
	}
	// next "kip"
	TRY(u, kip+1);
}

int main(int argc, char const *argv[])
{
	// std::string filename = "input.inp";
	input();
	std::cout << "\n\n##################################\n\n";
	// k = N (upper bound) in case each timeslot has only one subject
	// p.resize(N+1, std::vector<int> (N+1, 0));
	p.resize(N+1, -1);
	m.resize(N+1, std::vector<int> (M+1, 0));
	k = INT32_MAX;

	auto start = std::chrono::high_resolution_clock::now();
	TRY(1, 1);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Benchmark time: " << std::setprecision(4) << duration.count()/10002 << "\n";

	if (k != INT32_MAX) {
		std::cout << "The number of feasible solution: " << solution_count << "\n";
		std::cout << "Objective value: " << k << "\n";
	}
	else {
		std::cout << "No solution!\n";
	}

	return 0;
}

// 5
// 15 20 25 10 15
// 3
// 16 17 26
// 4
// 1 4
// 2 4
// 1 2
// 1 5

