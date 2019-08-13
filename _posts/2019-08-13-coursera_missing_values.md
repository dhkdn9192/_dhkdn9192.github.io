---
title: Coursera Kaggle - 3. Missing Values
---

![intro](../img/posts/20190730_coursera_kaggle_intro.png)



### Week1 - Feature preprocessing and generation with respect to models: Missing Values
- Coursera "How to Win a Data Science Competition: Learn from Top Kagglers" 강의 정리.
- Kaggle 문제 해결을 통한 Data Science 능력을 키워보자.
- https://www.coursera.org/learn/competitive-data-science


### 1. Overview

- Missing Values란?
  - not a number
  - very large numbers
  - \-1, -999, 999
  - empty string
  - etc...


### 2. Hidden NaNs

- 어떤 데이터가 missing value라는 것을 어떻게 알 수 있을까?
- 아래와 같은 데이터 예시에서 missing value가 있다고 가정할 경우, 연속균등분포(uniform distribution)을 통해 missing value가 \-1로 대치되었다고 가정할 수 있다.
![ddd](../img/posts/20190813_nan_uniform_disribution.png)

- missing values는 위와 같이 특정 값들로 대치되어 드러나지 않고 숨어있을 수 있다. 

### 3. Fillna approaches

1. \-999, \-1, etc
   - nan을 특정 고정 값으로 대치한다.
   - 선형 데이터에는 적합하지 않다.
2. mean, median
   - nan을 평균이나 중간값으로 대치한다.
   - 선형 데이터나 신경망 모델에서 적합한 방법이다.
3. reconstruct value 
   - 데이터를 새로 구성하여 채워넣는다.


### 4. Missing Values Reconstruction

- 아래와 같은 시계열 데이터에선 missing value를 추정하여 채워넣을 수 있다.
- 그러나 이는 매우 드문 예로, 대부분의 경우엔 sample들은 서로 독립적이다
![ee](../img/posts/20190813_nan_reconstruct_linear.png)