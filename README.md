### 리더보드 

- 대회URL : https://dacon.io/competitions/official/236013/leaderboard
- 최종 파일 : ./submission/
  
![image](https://github.com/user-attachments/assets/91a76f53-9199-47c6-bd02-be3739320f37)

--------------------


```
Fold  1 AUC : 0.713411
Fold  2 AUC : 0.680495
Fold  3 AUC : 0.708398
Fold  4 AUC : 0.704861
Fold  5 AUC : 0.714853
Full AUC score 0.704186

feature
ANONYMOUS_1_ANONYMOUS_2_div                                  113
FE_divide_FE_FE_woe_bin_median                               109
CU_divide_CU_YEAR_sum                                         96
ZN_minus_ZN_YEAR_median                                       89
YEAR_COMPONENT                                                83
CR_ANONYMOUS_1_plus                                           69
ANONYMOUS_1_ZN_divide_ANONYMOUS_1_ZN_YEAR_COMPONENT_max       66
ANONYMOUS_1_V40_divide_ANONYMOUS_1_V40_YEAR_sum               65
ANONYMOUS_2_NI_div                                            65
ZN_CR_div                                                     64
ANONYMOUS_1_ZN_divide_ANONYMOUS_1_ZN_YEAR_sum                 64
ANONYMOUS_1_MO_div                                            64
ANONYMOUS_1_NI_plus                                           63
ANONYMOUS_1_CU_plus                                           60
ANONYMOUS_1_CU_minus                                          53
ANONYMOUS_1_TI_plus                                           52
```

****

변수 생성 절차 
1. numeric combination feature 맨 처음에 실행
2. id, categorical combination feature 두번째로 실행 
3. group operations 맨 마지막에 실행 

****

변수선택

1. 상관관계 0.99 이상 제거 
2. n unique count 30 아래 제거 
3. psudo feature 만들어서, cv=5 vi가 psudo feature보다 아래인 변수 제거 

****

수치가 평균보다 낮을수록
불량기계건수가 증가 

낮을수록타겟증가 = ['CR', 'CO', 'H2O', 'MN', 'NI', 'TI', 'V']

****

변수중요도높은변수 = ['V40','FE','ZN','CU'] # 변수중요도 높은 원소들 <br>
타겟과음의상관관계 = ['ZN','H2O'] # 타겟과 마이너스 상관관계 <br>

제로비율이높은변수 = ['AG','CO','H2O','TI','V']+['NI']<br>
제로비율이절반변수 = ['CR','MN','MO']<br>
제로비율이낮은변수 = ['ANONYMOUS_1','ANONYMOUS_2','PQINDEX ','V40','FE','ZN','CU']<br>

****

Train Data 에서 AL Feature를 확인해보시면
AL 이 기준을 넘으면 다른 Feature에 관계 없이 모두 다 이상(Y=1)으로 판정되어 있습니다. (추정 기준 : 40~50 ppm)


오류 분석하기 (오류나는 건에 대해서 데이터 탐색)
-> 과적합 

train에만 있고 test에는 없는 변수로 파생변수 생성 
-> ??

id변수 찾기
-> ??

outlier 변수
(수치형변수 - 평균) 평균대비 1년에 몇번이나 수치가 올라갔는가?

****

ID : ID<br>
COMPONENT_ARBITRARY : sample 오일 관련 부품 (Component 4종)<br>
ANONYMOUS_1 : 무명 Feature 1, 수치형 데이터<br>
YEAR : 오일 진단 년도(Year)<br>
SAMPLE_TRANSFER_DAY : 오일 샘플링 후 진단 기관으로 이동한 기간(Days)<br>
ANONYMOUS_2 : 무명 Feature 2, 수치형 데이터<br>
AG : 은(Silver) 함유량<br>
AL : 알루미늄(Aluminium) 함유량<br>
B : 붕소(Boron) 함유량<br>
BA : 바륨(Barium) 함유량<br>
BE : 베릴륨(Beryllium) 함유량<br>
CA : 칼슘(Calcium) 함유량<br>
CD : 카드뮴(Cadmium) 함유량<br>
CO : 코발트(Cobolt) 함유량<br>
CR : 크로뮴(Chromium) 함유량<br>
CR : 구리(Copper) 함유량<br>
FH2O : 물(Water) 수치(By FT-IR)<br>
FNOX : 질소산화물(Nox) 수치(By FT-IR)<br>
FOPTIMETHGLY : 비식별화<br>
FOXID : 산화(Oxidation) 수치(By FT-IR)<br>
FSO4 : 황산염(S04) 수치(By FT-IR)<br>
FTBN : 염기성 첨가제물질 수치(By FT-IR)<br>
FE : 철(Iron) 함유량<br>
FUEL : 연료 함유량<br>
H2O : 물 함유량<br>
K : 칼륨(Potassium) 함유량<br>
LI : 리튬(Lithium) 함유량<br>
MG : 마그네슘(Magnesium) 함유량<br>
MN : 망가니즈(Manganese) 함유량<br>
MO : 몰리브덴(Molybdenum) 함유량<br>
NA : 나트륨(Sodium) 함유량<br>
NI : 니켈(Nickel) 함유량<br>
P : 인(Phosphorus) 함유량<br>
PB : 납(Lead) 함유량<br>
PQINDEX : 입자 정량화 지수(Particle Quantifier Index)<br>
S : 황(Sulphur) 함유량<br>
SB : 안티몬(Antimony) 함유량<br>
SI : 규소(Silicone) 함유량<br>
SN : 주석(Tin) 함유량<br>
SOOTPERCENTAGE : 그을음 정도<br>
TI : 티타늄(Titanium) 함유량<br>
U100 : 100㎛ 이상 입자 크기(Particle Count)<br>
U75 : 75㎛ 이상 입자 크기(Particle Count)<br>
U50 : 50㎛ 이상 입자 크기(Particle Count)<br>
U25 : 25㎛ 이상 입자 크기(Particle Count)<br>
U20 : 20㎛ 이상 입자 크기(Particle Count)<br>
U14 : 14㎛ 이상 입자 크기(Particle Count)<br>
U6 : 6㎛ 이상 입자 크기(Particle Count)<br>
U4 : 4㎛ 이상 입자 크기(Particle Count)<br>
V : 바나듐(Vanadium) 함유량<br>
V100 : 점도(Viscosity) @ 100 degrees<br>
V40 : 점도(Viscosity) @ 40 degrees<br>
ZN : 아연(Zinc) 함유량<br>
Y_LABEL : 오일 정상 여부(0 : 정상, 1 : 이상)<br>
