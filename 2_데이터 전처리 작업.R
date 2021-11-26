# 1. 데이터 전처리 패키지

# 1-2 데이터 전처리 패키지 유형

## 1-2-1 plyr 패키지
### 원본 데이터를 분석하기 쉬운 형태로 나눠서 다시 새로운 형태로 만들어 주는 패키지
### 데이터 분할(split), 원하는 방향으로 특정 함수 적용(apply), 그 결과를 재조합(combine)하여 반환
library(plyr)

### adply : 배열을 입력받아 함수를 적용한 후 결과를 데이터 프레임으로 반환
### adply(data(행렬, 배열, 데이터 프레임), margins(함수 적용 방향 지정 1=행, 2=열, c(1,2)=행과 열), fun(지정한 방향으로 적용할 함수))
adply(iris, 1, function(row) {row$Sepal.Length >= 5.0 & row$Species == 'setosa'})

### ddply : 데이터 프레임을 입력받아 함수를 적용한 후 데이터 프레임으로 결과 반환
### ddply(data(데이터 프레임), .variables(그룹화할 기준이 되는 변수), ddply-func(내부 함수), fun(.variables에 지정한 변수들의 값이 같은 데이터별로 적용할 함수))
ddply(iris, .(Species, Petal.Length < 1.5), function(sub){
  data.frame(
    mean_to = mean(sub$Sepal.Length), mean_so = mean(sub$Sepal.Width),
    mean_wo = mean(sub$Petal.Length), mean_jo = mean(sub$Petal.Width))
})

ddply(iris, .(Species), summarise, mean_to = mean(Sepal.Length))

### transform : 데이터 프레임에 새로운 변수 추가
### transform(_data, tag1(변수명) = value1(열 데이터), tag2 = value2, ...)
transform(iris, Total.w = Sepal.Width + Petal.Width)


## 1-2-2 dplyr 패키지
### 데이터 전처리 작업에 가장 많이 사용되는 패키지
library(dplyr)

### select : 데이터에서 원하는 변수의 특정 열만 추출
iris %>% select(Sepal.Length)

### filter : 원하는 조건에 따라서 데이터 필터링 추출
iris %>% filter(Species == 'setosa') %>% select(Sepal.Length, Sepal.Width)

### mutate : 기존 데이터에 파생변수 만들어 추가
iris %>% 
  filter(Species == 'virginica') %>%
  mutate(Len = ifelse(Sepal.Length>6, 'L', 'S')) %>% select(Species, Len)

### group_by : 지정한 변수들을 기준으로 데이터를 그룹화하는 함수
### summarise : 요약 통계치 계산하는 함수
iris %>%
  group_by(Species) %>%
  summarise(Petal.Width = mean(Petal.Width))

### arrange : 특정한 열을 기준으로 데이터 정렬
iris %>% 
  filter(Species == 'virginica') %>%
  mutate(Len = ifelse(Sepal.Length>6, 'L', 'S')) %>% 
  select(Species, Len, Sepal.Length) %>%
  arrange(desc(Sepal.Length))

### inner_join : 두 데이터 프레임에서 공통적으로 존재하는 모든 열 병합
### left_join : 왼쪽 데이터 프레임을 기준으로 모든 열 병합
### right_join : 오른쪽 데이터 프레임을 기준으로 모든 열 병합
### full_join : 두 데이터 프레임에 존재하는 모든 열 병합
X <- data.frame(Department = c(11, 12, 13, 14),
                DepartmentName = c("Production", "Sales", 
                                   "Marketing", "Research"),
                Manager = c(1, 4, 5, NA))

Y <- data.frame(Employee = c(1, 2, 3, 4, 5, 6),
                EmployeeName = c("A", "B", "C", "D", "E", "F"),
                Department = c(11, 11, 12, 12, 13, 21),
                Salary = c(80, 60, 90, 100, 80, 70))

inner_join(X, Y, by = "Department")
left_join(X, Y, by = "Department")
right_join(X, Y, by = "Department")
full_join(X, Y, by = "Department")

### bind_rows : 데이터의 행들을 이어 붙여 결합
#### 열 이름이 같지 않더라도 빈자리가 NA로 채워짐
x <- data.frame(x = 1:3, y = 1:3)
y <- data.frame(x = 4:6, z = 4:6)
bind_rows(x, y)

### bind_cols : 데이터의 열들을 이어 붙여 결합
#### 행 개수가 동일해야 하고, 동일하지 않으면 에러
x <- data.frame(title = c(1:5), a = c(30, 70, 45, 90, 65))
y <- data.frame(b = c(70, 65, 80, 80, 90))
bind_cols(x, y)


## 1-2-3 reshape2
library(reshape2)

### melt : 원데이터의 여러 변수명과 값이 행에 존재할 수 있도록 데이터 변환
#### melt(data, id.vars, measure.vars(지정하지 않으면 id 제외 나머지), na.rm)
#### 여러 변수로 이루어진 데이터를 id, variable, value 세 컬럼만으로 구성된 데이터로 변환
library(MASS)
a <- melt(data = Cars93, 
          id.vars = c("Type", "Origin"),
          measure.vars = c("MPG.city", "MPG.highway"))


### cast : 데이터의 모양을 다시 재조합
#### dcast : melt 함수로 변경된 형태의 데이터를 다시 기존처럼 여러 컬럼을 가진 형태로 변환
#### dcast(data, formula, fun.aggregate)
#### formula : id ~ variables / 아무 변수도 지정하지 않으려면 .를 지정
#### fun.aggregate : id를 기준으로 여러 행이 존재할 경우 해당 행들에 적용할 집합 함수
dcast(data = a, Type + Origin ~ variable, fun = mean)


## 1-2-4 data.table
### 연산속도가 매우 빨라서 크기가 큰 데이터를 처리하거나 탐색하는 데 효과적
install.packages('data.table')
library(data.table)

### 데이터 테이블 생성 : data.table(tag(변수명) = value(값), )
t <- data.table(x = c(1:3),
                y = c("가", "나", "다"))
 
### 데이터 테이블 변환 : as.data.table(데이터 프래임)
iris_table <- as.data.table(iris)
iris

### 데이터 접근 : data[행, 열, by]
iris_table[1, ]
iris_table[c(1:2),]
iris_table[, mean(Petal.Width), by = Species] #Species별 평균 계산


#-------------------------------------------------------------------------------


# 2. 데이터 정제

# 2-1 결측값
ds_NA <- head(airquality, 5)

is.na(ds_NA)
complete.cases(ds_NA) #행별로 결측값 확인

install.packages('mlbench')
library(mlbench)

data("PimaIndiansDiabetes2")
Pima2 <- PimaIndiansDiabetes2
sum(is.na(Pima2))
sum(!complete.cases(Pima2)) # 결측값이 있는 행의 수
colSums(is.na(Pima2)) # 각 컬럼별 결측값의 수 

## summary 함수를 이용하여 확인 가능


## 2-1-3 결측값 처리
### 결측값 삭제(특정 컬럼)
colSums(is.na(Pima2))
Pima2$insulin <- NULL
Pima2$triceps <- NULL
sum(!complete.cases(Pima2))
colSums(is.na(Pima2))

### 단순 대치법
#### 완전 분석법 : 결측값이 있는 특정 걸럼에 대한 행 삭제(is.na)와 결측값이 존재하는 모든 행 삭제(na.omit)
library(dplyr)
Pima3 <- Pima2 %>% filter(!is.na(glucose) & !is.na(mass))
colSums(is.na(Pima3))

Pima4 <- na.omit(Pima3)
colSums(is.na(Pima4))

#### 평균 대치법 : 데이터의 결측값을 평균값으로
#### 결측값이 존재할 경우에 삭제하지 않고 평균, 중위수 등으로 대체
head(ds_NA$Ozone)
ds_NA$Ozone <- ifelse(
  is.na(ds_NA$Ozone),
  mean(ds_NA$Ozone, na.rm = TRUE),
  ds_NA$Ozone)
table(is.na(ds_NA$Ozone))
ds_NA$Ozone

ds_NA2 <- head(airquality, 5)

ds_NA2[
  is.na(ds_NA2$Ozone), "Ozone"] <-
  mean(ds_NA2$Ozone, na.rm = TRUE)

table(is.na(ds_NA2$Ozone))

summary(Pima3)
mean_press <- mean(Pima3$pressure, na.rm = TRUE)
std_before <- sd(Pima3$pressure, na.rm = TRUE)
Pima3$pressure <- ifelse(is.na(Pima3$pressure), mean_press, Pima3$pressure)
std_after <- sd(Pima3$pressure)
std_after - std_before


# 2-2 이상값

## 2-2-2 이상값 판별
### ESD(Extreme Studentized Deviation) : 평균에서 3 표준편차보다 큰 값들 판별
#### 평균을 이용하기 때문에 이상값에 민감함
score <- c(1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 100000000)
name <- c("A", "B", "C", "D", "E",
          "F", "G", "H", "I", "J", "K", "L")
df_score <- data.frame(name, score)

esd <- function(x){
  return(abs(x - mean(x))/ sd(x) < 3)
}
library(dplyr)

df_score %>% filter(esd(score))

### 사분위수 이용
#### 이상치에 민감하지 않음
#### 박스플롯 이용
score <- c(65, 60, 70, 75, 200)
name <- c("Bell", "Cherry", "Don", "Jake", "Fox")
df_score <- data.frame(name, score)
box_score <- boxplot(df_score$score)

box_score$out # out변수를 이용한 이상값 확인
box_score$stats

min_score <- box_score$stats[1]
max_score <- box_score$stats[5]

df_score %>% filter(score >= min_score & score <= max_score)

#### IQR 함수 이용
min_score <- median(df_score$score) - 2 * IQR(df_score$score)
max_score <- median(df_score$score) + 2 * IQR(df_score$score)

df_score %>% filter(score >= min_score & score <= max_score)


## 2-2-3 이상값 처리
### 삭제 : 추정치의 분산은 작아지지만, 실제보다 과소(과대) 추정되어 편의 발생할 수 있음
### 대체 : 하한값보다 작으면 하한값으로 대체, 상한값보다 크면 상한값으로 대체
### 변환 : 자연로그 취하기
### 분류 : 서로 다른 그룹으로 통계적인 분석 실행하여 처리


# 2-3 데이터 변환

## 2-3-3 데이터의 범위 변환
### scale(x(행렬), center = , scale = ) x-center(min) / scale(max - min)
### default : center = x평균, scale = x표준편차
### 최소-최대 정규화
data <- c(1, 3, 5, 7, 9)
data_minmax <- scale(data, center = 1, scale = 8)

a <- 1:10
normalize <- function(a) {
  return((a-min(a)) / (max(a) - min(a)))
}
normalize(a)
as.vector(scale(a)) #scale 함수 이용하여 표준화

### z-score
data_zscore <- scale(data)
mean(data_zscore)
sd(data_zscore)


# 2-4 표본 추출 및 집약처리

## 2-4-1 표본 추출
### sample : base패키지의 함수(단순 무작위 추출)
### sample(x, size, replace(True : 복원, FALSE : 비복원), prob(가중치))
s <- sample(x = 1:10, size = 5, replace = FALSE)
s <- sample(x = 1:10, size = 5, replace = TRUE)
s <- sample(x = 1:10, size = 5, replace = TRUE, prob = 1:10) # 1에서 10까지 각각 가중치

### createDataPartition : caret 패키지. 특정 비율로 훈련, 평가 데이터로 랜덤하게 분할(층화 추출)
### createDataPartition(y, times(분할 수), p(훈련데이터 비율), list(결과를 리스트로 반환할지))
library(caret)

ds <- createDataPartition(
  iris$Species, times = 1, p = 0.7)

table(iris[ds$Resample1, "Species"])

table(iris[-ds$Resample1, "Species"])

idx <- as.vector(ds$Resample1)
ds_train <- iris[idx, ]
ds_test <- iris[-idx]

### createFolds : k-fold
### create(y, k, list(결과를 리스트로 반환할지), returnTrain(list = TRUE일 경우 사용, TRUE : 복원 / FALSE : 비복원))
k_fold <-
  createFolds(iris$Species,
              k = 3,
              list = TRUE,
              returnTrain = FALSE)
k_fold


## 2-4-2 기초 통계량 추출
### 최빈수 : 직접 함수 정의하여 구하기
getmode <- function(x) {
  y <- table(x)
  names(y)[which(y == max(y))]
}

x <- c(2,1,1,3,1)
getmode(x)

### 순위 게산
#### 데이터의 행을 집약한 후에 집약한 결과를 다시 계산하여 각 해ㅐㅇ에 첨부

x <- c(1, 1, 5, 5, 9, 7)
library(dplyr)

row_number(x) # 1 2 3 4 6 5
min_rank(x) # 1 1 3 3 6 5
dense_rank(x) # 1 1 2 2 4 3

cars %>% 
  arrange(dist) %>% 
  mutate(rank = row_number(dist))

cars %>% 
  arrange(dist) %>% 
  mutate(rank = min_rank(dist))

cars %>% 
  arrange(dist) %>% 
  mutate(rank = dense_rank(dist))