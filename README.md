# ECG-categorization-with-DNN
#### (Caution!)this file is for working on colab. if you use this code, please paste this on colab
##### number of date : 87552
##### number of columns : 187
##### Used model : Sequentail
##### number of categorization(output) : 5(0~4)
##### Accuracy : 100%
##### 프로젝트를 진행한 이유:
##### 요즘 스마트 워치들이 점점 심장 박동 출력 기능도 지원하고 있다. 물론 병원에서 찍는 ECG만큼 정확하지 않지만, 헬스케어가 더이상 병원에서만 이루어지는 것이 아닌 우리 삶속에 녹아드는 시대의 흐름에 맞추어 이 프로젝트를 진행하였다.
##### 프로젝트를 진행하면서 느낀 점:
##### 1. softmax 사용법
##### 2. hyperparameter 조정
##### 3. 역시나 양질의 데이터의 중요성을 느꼈다. acc 가 100%를 찍었지만, 이 data는 캐글에서 제공한 data이다. data가 없으면 딥러닝을 할 수 없고, 이런 양질의 의료 데이터는 병원에서 쉽게 내주지 않는다. 따라서 의료기관과 협력하여 data를 가져오는 플랫폼이 활성화되는 것이 의료에 있어서 ai를 적용하는 중요한 첫걸음이 될 것이다.
##### 프로젝트를 진행하면서 느낀 아쉬운 점:
##### 카테고라이징이 5가지 밖에 없었던 것이 가장 아쉽다. ECG는 cardio 계에 문제가 있는 사람들의 치료계획을 수립할 때, 중요하게 쓰이는 data 인 만큼 이를 통해 다양한 진단을 할 수 있을 것 같은다. 더 다양한 분류가 있었으면 소비자들에게 더 많은 정보를 제공할 수 있을 것 같다.
