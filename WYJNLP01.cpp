#include <iostream>
#include <CString>
#define MAX_LENGRH 100
using namespace std;

int matrix[][4] = {{-1, 1, 0, -1}, {1, -1, -1, 0}, {0, -1, -1, 1}, {-1, 0, 1, -1}};

int main() {
    char s[MAX_LENGRH];
    int flag;
    cout << "请输入被检测字符串：...\n";
    while(scanf("%s", &s) != EOF) {
        //cout << s << strlen(s) << endl;
        int lie = 0;
        for(int i = 0; i < strlen(s); i++) {
            flag = 0;
            for(int j = 0; j < 4; j++) {
                if((s[i] - '0') == matrix[j][lie]) {
                    lie = j;
                    flag = 1;
                    break;
                }
            }        
			
			if(!flag)
            	break;
        }
        
		if(!flag) cout << "字符串中的字符不在输入字符集中！\n";
		else {
			if(lie)
            	cout << "字符串不可以被有限自动机接收！\n";
        	else
				cout << "字符串可以被有限自动机接收！\n";
		} 

    }
    //cout << matrix[1][2] << endl;
    return 0;
}
