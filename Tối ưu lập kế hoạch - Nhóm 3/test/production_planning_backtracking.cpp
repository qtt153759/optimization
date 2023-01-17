#include<iostream>
using namespace std;

int N; // Số lượng sản phẩm
int A; // Diện tích thửa ruộng
int C; // Chi phí tối đa có thể sử dụng
int m[100]; // Số lượng đơn vị sản phẩm tối thiểu cần sản xuất ứng với 1 sản phẩm i
int a[100]; // Diện tích đất để tạo ra 1 đơn vị sản phẩm i
int c[100]; // Chi phí để tạo ra 1 đơn vị sản phẩm i
int f[100]; // Lợi nhuận thu được từ 1 đơn vị sản phẩm i
int k[100]; // Số đơn vị sản phẩm i có thể sản xuất tối đa
int result[100]; // Kết quả tạm thời
int land = 0; // Diện tích đất đang xét
int cost = 0; // Chi phí đang xét
int profit = 0; // Lợi nhuận đang xét
int best_result[100]; // Số lượng đơn vị sản phẩm ứng với sản phẩm i tối ưu
int best = 0; // Lợi nhuận tối ưu

void TRY(int x){
    for(int i = m[x]; i <= k[x]; i++){
        land += i*a[x];
        cost += i*c[x];
        if(land > A || cost > C){
            land -= i*a[x];
            cost -= i*c[x];
        }
        else{
            profit += i*f[x];
            result[x] = i;
            if(x==N){
                if(profit>best){
                    best = profit;
                    for(int j = 1; j<=N; j++){
                        best_result[j]=result[j];
                    }
                }
            }
            else{
                TRY(x+1);
            }
            land -= i*a[x];
            cost -= i*c[x];
            profit -= i*f[x];
        }
    }
}

int main(){
    cin >> N >> A >> C;

    for(int i = 1; i<=N; i++){
        cin >> c[i];
    }

    for(int i = 1; i<=N; i++){
        cin >> a[i];
    }

    for(int i = 1; i<=N; i++){
        cin >> f[i];
    }

    for(int i = 1; i<=N; i++){
        cin >> m[i];
    }

    int Ka, Kc;
    for(int i = 1; i<=N; i++){
        Ka=0;
        Kc=0;
        for(int j = 1; j<=N; j++){
            if (i!=j){
                Ka+=a[j]*m[j];
                Kc+=c[j]*m[j];
            }
        }
        k[i] = int(min((A-Ka)/a[i], (C-Kc)/c[i]));
    }

    for(int i = 1; i<=N; i++){
        cout << k[i] << " ";
    }
    cout<<endl;

    TRY(1);
    cout << "Kết quả tốt nhất " << best << endl;
    cout << "Số đơn vị sản phẩm tối ưu " << endl;
    for(int i=1; i<=N; i++){
        cout<<best_result[i]<<" ";
    }
}

