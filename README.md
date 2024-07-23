"#k_7MachineLearning" 
1-1)환경설정
pyhton 3.11.x

install matplotlib scikit-learn numpy pandas

vscode
open folder-c드라이브 내 폴더 선택
확장플러그인-python, python debugger(최종 .py 파일 실행을 위해서)
	     jupyter,jupyter keymap, jupyter slide show, jupyter cell tags, jupyter notebook renderers (ML의 경우, 중간과정을 확인해야해서)

1-2) vscode에서 그래프 출력을 위한 환경 설정
#라이브러리 먼저 선언하고 preamble를 뒤에 써줘야 함
import pandas as pd
import mglearn 
import preamble

#Graphviz 2.38 설치 후 "런타임 오류: Graphviz 실행 파일이 시스템 경로에 있는지 확인하세요
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

#트리가 안 보일 때 ModuleNotFoundError: No module named 'graphviz'
https://graphviz.org/download/  graphviz-12.0.0(64-bit) EXE installer (all users)
터미널에서 pip install graphviz

#폰트 설정 다시 해야함
윈도우, 리눅스, macOS 전부 동일
pip install git+https://github.com/sigmadream/koreanize-matplotlib.git
import koreanize_matplotlib(preamble.py에 추가)
기존의 한국어 설정 주석 처리
# plt.rcParams['font.family']='Malgun Gothic'
# plt.rcParams['font.size']=9


2)지도학습-기울기 구하기
  
3) 문제와 데이터 이해하기
문제 정의-문제를 보고 회귀로 풀지, 분류로 풀지
데이터-숫자로 데이터 표기
문제가 주어졌을 때, 빠르게 분석하는 것이 이 책의 교육과정의 목표이다.

4) 머신 러닝
데이터 
모델(학습)
예측
검증


5) 독립변인, X, 특성
	data, 종속변인 , y, 타겟

	data가 딕셔너리일 때,
	iris_database.keys()
	#'data'                샘플       
	#'feature_names'       특징
	#'target_names'        답안지 (붓꽃 3종류)

	iris_database['feature_names']
	iris_database['data'].shape
	iris_database['target_names'] 

6) train, test 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(iris_database['data'], iris_database['target'], random_state=42)

*random_state=0 또는 42로 설정 시, 똑같은 데이터셋이 나온다. 교육용일 떄는 이미 완성된 알고리즘이기에 확인하려고 코드 돌리는 것이니 random_state 설정한다.
하지만 프로젝트 할 때나 최적의 알고리즘을  찾아야 하는 경우에는 , random_state를 지워야 함.

7)과대적합 : 새로운 데이터에 일반화 되기 어렵다/문제 발생 시 해결법이 없다

  과소적합 : 문제 발생 시 더 많은 데이터를 통해 해결한다.

  *선형,이웃 알고리즘은 파라미터가 작을 때 주로 쓰며, 이 알고리즘들은 단순하기에 과대적합이 일어날 가능성이 높다
  =======================
그리드서치-스케일링(X_train , X_test  둘다 스케일링)

파이프라인 -MAKE PIPELINES(이름이 자동으로 정해진다/여러개의 파이프라인 만들어 연결) - 서치

sklearn에서 성분을 보기 위해서 _ 끝에 적어야 함 (예-pipe_short.named_steps["pca"].components_)

pd.read_csv(data_url,sep=r"\s+" ,skiprows=22 , header=None)

# 보스턴 주택 데이터를 평균으로 StandardScaler() 한다.  정규분포 그래프 (종 모양 그래프)
# 평균으로 할 시에 평균특성이 사그라드므로 특성을 키우고자 PolynomialFeatures 한다. 이차곡선그래프(U)
# 선형으로 선을 그어 회귀한다.Ridge()  선형그래프(/) =>보스턴 집값이 매우큰쪽과 매우작은쪽은 회귀의 정확도가 떨어질 것이다.
pipe =make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())