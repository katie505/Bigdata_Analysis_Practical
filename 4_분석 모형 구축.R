# 1. 통계 분석

# 1-1 주성분 분석

## 1-1-3 주성분 분석 구축 프로세스
iris_pca <- princomp(iris[, -5],
                     cor = FALSE, #TRUE : 상관, FALSE : 공분산
                     scores = TRUE)

summary(iris_pca)

### 적재값을 바탕으로 각 주성분을 기존 변수들의 선형결합으로 나타낼 수 있음
### 적재값이 클수록 각 주성분과 변수 간의 관련성이 높음
iris_pca$loadings # 각 주성분의 적재값 확인 가능

### 각 주성분들을 축으로 하여 계산된 행별 좌표
iris_pca$scores

plot(iris_pca, type = "l", main = "iris 스크리 산점도")

### x축을 제 1주성분, y축을 제 2주성분으로 하고, 각 변수에 대한 주성분 적재값을 화살표로 시각화
### 주성분에 영향을 미칠수록 수평
biplot(iris_pca, scale = 0, main = "iris biplot")


# 1-2 요인 분석
## 요인 분석에서 요인을 추출하는 방법 중 가장 널리 사용되는 방식이 주성분 분석
## 프로세스 : 데이터 입력 -> 상관계수 산출 -> 요인 추출 -> 요인 적재량 산출
## -> 요인 회전 -> 생성된 요인 해석 -> 요인 점수 산출


#-------------------------------------------------------------------------------


# 2. 정형 데이터 분석

# 2-1 분류 모델

## 2-1-1 회귀 분석
### 단순 선형 회귀 수행
install.packages('ISLR')

library(ISLR)

#### 보통 결정계수가 0.65보다 크면 적합한 모형
summary(lm(Salary ~ PutOuts, data = Hitters))

### 다중 선형 회귀 분석
str(Hitters)
head(Hitters)
summary(Hitters)

hitters <- na.omit(Hitters)
summary(hitters)

full_model <- lm(Salary ~., data = hitters)

summary(full_model)

#### full_model에서 유의한 변수들만 선택
first_model <- lm(Salary ~ AtBat + Hits + Walks + CWalks + Division + PutOuts, data = hitters)

fit_model <- step(first_model, direction = 'backward')

install.packages('car')
library(car)
vif(fit_model)

second_model <- lm(Salary ~ Hits + CWalks + Division + PutOuts, data = hitters)
vif(second_model)

summary(second_model)


## 2-1-2 로지스틱 회귀 분석
str(Default)

head(Default)
summary(Default)

bankruptcy <- Default

set.seed(202012)

train_idx <- sample(1:nrow(bankruptcy), 
                    size = 0.8*nrow(bankruptcy), 
                    replace = FALSE)
test_idx <- (-train_idx)

bankruptcy_train <- bankruptcy[train_idx, ]
bankruptcy_test <- bankruptcy[test_idx, ]

full_model <- glm(default ~ ., family = binomial, data = bankruptcy_train)

step_model <- step(full_model, direction = "both")

summary(step_model)

null_deviance <- 2354.0
residual_deviance <- 1287.4
model_deviance <- null_deviance - residual_deviance

pchisq(model_deviance, df = 2, lower.tail = FALSE)
 
vif(step_model)

pred <- predict(step_model,
                newdata = bankruptcy_test[, -1],
                type = 'response')

df_pred <- as.data.frame(pred)

df_pred$default <- ifelse(df_pred$pred >= 0.5, 'Yes', 'No')

df_pred$default <- as.factor(df_pred$default)

library(caret)
confusionMatrix(data = df_pred$default, 
                reference = bankruptcy_test[,1])

library(ModelMetrics)
auc(actual = bankruptcy_test[,1], predicted = df_pred$default)


## 2-1-3 의사결정나무
### 연속적으로 발생하는 의사결정 문제를 시각화해서 의사결정이 이루어지는 시점과 성과 파악 쉽게 해줌
str(iris)
summary(iris)

library(rpart)

md <- rpart(Species ~., data = iris)

plot(md, compress = TRUE, margin = 0.5) # 트리 모형 시각화
text(md, cex = 1) # 텍스트 크기 조정

library(rpart.plot)

prp(md, type = 2, extra = 2) #의사결정나무 모형 시각화

ls(md)

### CP(복잡성), nsplit(가지의 분기 수), rel error(오류율), xerror(교차 검증 오류), xstd(교차 검증 오류의 표준오차)
### 가지치기 및 트리의 최대크기를 조절하기 위해 사용(교차 타당성 오차 확인)
md$cptable 

plotcp(md) # 교차 타당성 오차 확인

tree_pred <- predict(md,
                     newdata = iris,
                     type = "class")

library(caret)

confusionMatrix(tree_pred,
                reference = iris$Species)


## 2-1-4 서포트 벡터 머신(지도 학습 기반의 이진 선형 분류)
### 데이터를 분리하는 초평면 중에서 데이터들과 거리가 가장 먼 초평면 선택하여 분리
### 기계 학습의 한 분야로 사물 인식, 패턴 인식, 손글씨, 숫자 인식 등에서 활용됨
###  최대 마진을 가지는 비확률적 선형 판별에서 기초한 이진 분류기

library(e1071)

model <- svm(Species ~., data = iris)

pred <- predict(model, iris)
library(caret)
confusionMatrix(data = pred, reference = iris$Species)


## 2-1-5 K-NN(K-최근접 이웃)
### knn(train, test, cl(훈련 데이터의 종속변수), k)
### 새로운 데이터 클래스를 해당 데이터와 가장 가까운 k개 데이터들의 클래스로 분류
### 근접 이웃의 수 K는 학습의 난이도와 훈련 데이터의 개수에 따라 결정(일반적으로 훈련 데이터 개수의 제곱근)
data <- iris[, c('Sepal.Length',
                 'Sepal.Width', 'Species')]
set.seed(1234)

idx <- sample(x = c('train', 'valid', 'test'),
              size = nrow(data),
              replace = TRUE, prob = c(3, 1, 1))

train <- data[idx == 'train',]
valid <- data[idx == 'valid',]
test <- data[idx == 'test',]

train_x <- train[, -3]
valid_x <- valid[, -3]
test_x <- test[, -3]

train_y <- train[, 3]
valid_y <- valid[, 3]
test_y <- test[, 3]

### k를 변경하면서 분류 정확도가 가장 높은 k 탐색
library(class)


accuracy_k <- NULL


for(i in c(1:nrow(train_x))){
  
  set.seed(1234)
  knn_k <- knn(train = train_x,
               test = valid_x,
               cl = train_y,
               k = i)
  accuracy_k <- c(accuracy_k,
                  sum(knn_k == valid_y) / length(valid_y))

}

valid_k <- data.frame(k = c(1:nrow(train_x)),
                      accuracy = accuracy_k)

plot(formula = accuracy ~ k,
     data = valid_k,
     type = 'o',
     pch = 20,
     main = 'validation - optimal k')

min(valid_k[valid_k$accuracy %in%
              max(accuracy_k), 'k']) # 분류 정확도가 가장 높으면서 가장 작은 k

max(accuracy_k)

knn_13 <- knn(train = train_x,
              test = test_x,
              cl = train_y, k = 13) # 모형평가를 위해 테스트 데이터 적용

library(e1071)
confusionMatrix(knn_13, reference = test_y)


## 2-1-6 ANN(인공신경망)
### 입력값을 받아서 출력값을 만들기 위해 활성화 함수 사용
library(nnet)

iris.scaled <- cbind(scale(iris[-5]), iris[5])

set.seed(1000)

index <- c(sample(1:50, size = 35),
           sample(51:100, size = 35),
           sample(101:150, size = 35))

train <- iris.scaled[index, ]
test <- iris.scaled[-index,]

set.seed(1234)
model.nnet <- nnet(Species~.,
                   data = train,
                   size = 2, # 은닉층 
                   maxit = 200, # 반복할 최대 횟수
                   decay = 5e-04) # 가중치 감소의 모수

summary(model.nnet)


## 2-1-7 나이브 베이즈 기법
### 특성들 사이의 독립을 가정하는 베이즈 정리를 적용한 확률 분류기
### subset : 분석 데이터에서 훈련 데이터 선정 가능
### laplace(라플라스 추정기) : 중간에 0이 들어가서 모든 확률을 0으로 만들어 버리는 것을 방지하기 위해
library(e1071)
train_data <- sample(1:150, size = 100)

naive_model <- naiveBayes(Species ~.,
                          data = iris,
                          subset = train_data)

naive_model

library(caret)
pred <- predict(naive_model, newdata = iris)
confusionMatrix(pred, reference = iris$Species)


## 2-1-8 앙상블
### 여러 가지 동일한 종류 또는 서로 상이한 모형들의 예측/분류 결과를 종합하여 최종적인 의사결정에 활용

### 배깅 : 훈련 데이터에서 다수의 부트스트랩 자료를 생성하고, 각 자료를 모델링한 후 결합하여 최종 에측 모형 만듦
library(adabag)
library(mlbench)
data(PimaIndiansDiabetes2)
PimaIndianDiabetes2 <- na.omit(PimaIndiansDiabetes2)

train.idx <- sample(1:nrow(PimaIndianDiabetes2),
                    size = nrow(PimaIndianDiabetes2)*2/3)

train <- PimaIndianDiabetes2[train.idx,]
test <- PimaIndianDiabetes2[-train.idx,]

library(ipred)
md.bagging <- bagging(diabetes ~.,
                      data = train,
                      nbagg = 25)
md.bagging

pred <- predict(md.bagging, test)

library(caret)
confusionMatrix(as.factor(pred),
                reference = test$diabetes,
                positive = 'pos')

### 부스팅 : 예측력이 약한 모형들을 결합하여 강한 예측 모형 만듦
#### XGBoost(eXtreme Gradient Boosting) : 병렬처리가 지원되도록 구현하여 훈련과 분류 속도가 빠른 알고리즘
#### 훈련 데이터는 xgb.DMatrix 객체를 이용하여 생성

library(xgboost)

train.label <- as.integer(train$diabetes) - 1 # 종속변수를 0부터 시작하도록 1을 뺌
mat_train.data <- as.matrix(train[, -9]) # 훈련데이터를 xgb.DMatrix로 변환하기 위해 행렬로 변환
mat_test.data <- as.matrix(test[, -9])

xgb.train <- xgb.DMatrix(
  data = mat_train.data,
  label = train.label)

xgb.test <- xgb.DMatrix(
  data = mat_test.data)

param_list <- list(
  booster = 'gbtree', 
  eta = 0.001, # 학습률, 작을수록 과대 적합에 강건
  max_depth = 10, #한 트리의 최대 깊이
  gamma = 5, # Information Gain에 패널티 부여. 클수록 트리의 깊이가 줄어서 보수적임
  subsample = 0.8, # 훈련 데이터의 샘플 비율
  colsample_bytree = 0.8, # 개별 트리 구성할 때 컬럼의 subsample 비율
  objective = 'binary:logistic', # 목적 함수 지정
  eval_metric = 'auc') # 모델의 평가 함수

md.xgb <- xgb.train(
  params = param_list, 
  data = xgb.train,
  nrounds = 200, # 최대 부스팅 반복 횟수
  early_stopping_rounds = 10, # AUC가 10회 증가하지 않을 경우 학습 조기 중단
  watchlist = list(val1 = xgb.train), # 모형의 성능을 평가하기 위하여 사용하는 xgb.DMatrix 이름
  verbose = 1
)

xgb.pred <- predict(md.xgb,
                    newdata = xgb.test)

xgb.pred2 <- ifelse(xgb.pred >= 0.5,
                    xgb.pred <- 'pos',
                    xgb.pred <- 'neg')
xgb.pred2 <- as.factor(xgb.pred2)
library(caret)
confusionMatrix(xgb.pred2,
                reference = test$diabetes,
                positive = 'pos')


### 랜덤 포레스트 : 의사결정나무의 특징인 분산이 크다는 점을 고려하여 배깅과 부스팅보다 더 많은 무작위성을 주어 약한 학습기들을 생성한 후 이를 선형 결합하여 최종 학습기 만듦
#### 랜덤 입력에 따른 여러 트리의 집합인 포레스트를 이용한 분류
library(randomForest)
md.rf <- randomForest(diabetes ~., 
                      data = train,
                      ntree = 100, # 사용할 트리의 수
                      proximity = TRUE) # 근접측정
print(md.rf) # 모델에 사용되지 않은 데이터를 통한 에러 추정치 : 18.77%

#### importance 함수를 이용하여 변수의 중요도 알 수 있음
importance(md.rf)

pred <- predict(md.rf, newdata = test)
confusionMatrix(as.factor(pred),
                test$diabetes,
                positive = 'pos')


# 2-2 군집 모델

## 군집 분석 : 관측된 여러 개의 변숫값으로부터 유사성에만 기초하여 n개의 군집으로 집단화하고, 형성된 집단의 특성으로부터 관게를 분석하는 다변량 분석기법
## 군집의 결과는 계통도 또는 덴드로그램의 형태로 결과가 주어지며 각 개체는 하나의 군집에만 속하게 됨
str(USArrests)
head(USArrests)
summary(USArrests)

### 유클리디안 거리 측정
US.dist_euclidean <- dist(USArrests, 'euclidean')

### 맨하탄 거리 측정
US.dist_manhattan <- dist(USArrests, 'manhattan')

### 마할라노비스 거리 측정
mahalanobis(USArrests, colMeans(USArrests), cov(USArrests))

### 계층적 군집 분석(최단 거리법)
US.single <- hclust(US.dist_euclidean^2, method = 'single')
plot(US.single)

### 최장거리법
US.complete <- hclust(US.dist_euclidean^2, method = 'complete')
plot(US.complete)

### 군집 분석을 통한 그룹 확인
group <- cutree(US.single, k = 6)

rect.hclust(US.single, k = 6, border = 'blue')

### 비계층적 군집 분석(k-평균 군집)
#### k개만큼 원하는 군집 수를 초깃값으로 지정하고, 각 개체를 가까운 초깃값에 할당하여 군집을 형성하고 각 군집의 평균을 재계산하여 초깃값 갱신
#### 갱신과정을 반복하여 k개의 최종군집 형성
library(rattle)

df = scale(wine[-1])

set.seed(1234)

library(NbClust)
fit.km <- kmeans(df, 3, nstart = 25) # 총 25번을 시행하고 그중 최솟값을 취해라
fit.km$size # k개의 점과 가장 가까운 데이터의 개수
fit.km$centers # 속성별 k개의 점에 대한 위치

plot(df, col = fit.km$cluster)
points(fit.km$centers, col = 1:3,
       pch = 8, cex = 1.5)


# 2-3 연관 모델

## 연관성 분석 : 데이터 내부에 존재하는 항목 간의 상호 관계 혹은 종속 관계 찾아내기
library(arules)

## aprori : 트랜잭션 데이터를 다루고 데이터셋 내에서 최소 N번의 트랜잭션이 일어난 아이템 집합들을 찾아 연관 분석 수행

## transactions 데이터 변환 사례
mx.ex <- matrix(
  c(1,1,1,1,10,
    1,1,0,1,0,
    1,0,0,1,0,
    1,1,1,0,0,
    1,1,1,0,0
    ), ncol = 5, byrow = TRUE)

rownames(mx.ex) <-
  c('p1', 'p2', 'p3', 'p4', 'p5')

colnames(mx.ex) <-
  c('a', 'b', 'c', 'd', 'e')

### as(data, class(객체를 변경할 클래스 이름)) : 객체들을 리스트로 결합해주는 함수
trx.ex <- as(mx.ex, 'transactions')

summary(trx.ex)

inspect(trx.ex) 3

## 연관성 분석 수행
data("Groceries")

summary(Groceries)

apr <- apriori(Groceries,
               parameter = list(support = 0.01, # 최소 지지도
                                confidence = 0.3)) # 최소 신뢰도

inspect(sort(apr, by = 'lift')[1:10])
