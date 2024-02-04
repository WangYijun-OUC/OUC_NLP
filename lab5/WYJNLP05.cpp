#include <iostream>
#include <CString>
#define MAX_LENGRH 100
using namespace std;

char matrix[12][12] = {0};

void initMatrix() {
    matrix[0][1] = 'h';
    matrix[1][2] = 'a';
    matrix[2][3] = 'p';
    matrix[3][4] = 'p';
    matrix[4][5] = 'y';
    matrix[4][10] = 'y';
    matrix[5][6] = 'y';
    matrix[6][7] = '+';
    matrix[7][8] = 'e';
    matrix[8][9] = 'r';
    matrix[8][11] = 's';
    matrix[11][12] = 't';
}

void PrintMatrix() {
    for(int i = 0; i < 13; i++) {
        for(int j = 0; j < 13; j++) {
            cout << matrix[i][j] << "  ";
        }
        cout << endl;
    }
}

int main() {
    char s[MAX_LENGRH];
    int flag;
    initMatrix();
    cout << "ÇëÊäÈë¼ì²â×Ö·û´®...\n";
    
	while(scanf("%s", &s) != EOF) {
		flag = 0;
		if(!strcmp(s, "happy"))
			flag = 1;
		else if(!strcmp(s, "happier"))
			flag = 1;
		else if(!strcmp(s, "happiest"))
			flag = 1;
	
		if(!flag) {
			cout << "¸ñÊ½´íÎó£¡" << endl;
		}
			
		// PrintMatrix();

		if(flag) {
			cout << "happy" << "->";

			int line = 0;
			int tmp = 0;
			int flag1 = 0;
			int breakCircle = 0;

			for(int j = line + 1; j < 13; j++) {
				if(breakCircle)
					break;
				for(int k = tmp; k < strlen(s); k++) {
					if(s[k] == 'i' || flag1) {
						flag1 = 1;
						if(line == 4) {
							line++;
							break;
						}
						
						cout << matrix[line][j];
						line++;
						if(line == 8) {
							flag1 = 0;
							tmp = tmp + 2;
						}
						break;
					}
					else if(s[k] == matrix[line][j]) {
						cout << matrix[line][j];
						if(j == 10 || j == 9 || j == 12) {
							breakCircle = 1;
							break;
						}
						line = j;
						tmp++;
						break;
					}
					else{
						tmp = k;
						break;
					}
				}
			} 
		}
		
		cout << endl;
	}

    //cout << matrix[1][2] << endl;
    return 0;
}
