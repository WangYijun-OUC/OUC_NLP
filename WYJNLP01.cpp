#include <iostream>
#include <CString>
#define MAX_LENGRH 100
using namespace std;

int matrix[][4] = {{-1, 1, 0, -1}, {1, -1, -1, 0}, {0, -1, -1, 1}, {-1, 0, 1, -1}};

int main() {
    char s[MAX_LENGRH];
    int flag;
    cout << "�����뱻����ַ�����...\n";
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
        
		if(!flag) cout << "�ַ����е��ַ����������ַ����У�\n";
		else {
			if(lie)
            	cout << "�ַ��������Ա������Զ������գ�\n";
        	else
				cout << "�ַ������Ա������Զ������գ�\n";
		} 

    }
    //cout << matrix[1][2] << endl;
    return 0;
}
