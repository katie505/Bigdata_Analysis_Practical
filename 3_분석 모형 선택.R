# 1. 데이터 탐색

# 1-2 데이터 탐색 방법

## 1-2-1 개별 데이터 탐색
### 범주형 데이터
#### 빈도수 탐색
table(mtcars$cyl)

#### 백분율 및 비율 탐색
cnt <- table(mtcars$cyl)
total <- length(mtcars$cyl)
prop.table(cnt)
cnt / total

#### 시각화
barplot(cnt,
        xlab = "기통",
        ylab = "수량",
        main = "기통별 수량")

pie(cnt,
    main = "기통별 비율")

### 수치형 데이터
#### 요약 통계량
summary(mtcars$wt)

head(mtcars$wt)

str(mtcars)

#### 시각화
wt_hist <- hist(mtcars$wt,
                breaks = 5,
                xlab = "무게",
                ylab = "수량",
                main = "무게별 수량")

wt_box <- boxplot(mtcars$wt,
                  main = "무게 분포")


## 1-2-2 다차원 데이터 탐색
### 범주형 - 범주형 : 빈도수와 비율 활용
table(mtcars$am, mtcars$cyl)
mtcars$label_am <- factor(mtcars$am, 
                          labels = c("automatic", "manual"))
table(mtcars$label_am, mtcars$cyl)

prop_table <- prop.table(table(mtcars$label_am, mtcars$cyl)) * 100 # 백분율 구하기

#### addmargins(x, margin(합계를 표시하려는 행 또는 열))
#### digits = 1 : 소수 1번째자리까지
addmargins(round(prop_table, digits = 1)) # 각 열과 행에 대한 백분율의 합계

#### 시각화
barplot(table(mtcars$label_am, mtcars$cyl),
        xlab = "실린더수",
        ylab = "수량")

### 수치형 - 수치형 : 상관계수(상관관계), 산점도(상관성 시각화)
#### 피어슨 상관계수
cor_mpg_wt <- cor(mtcars$mpg, mtcars$wt)

plot(mtcars$mpg, mtcars$wt)

### 범주형 - 수치형 : 범주형 데이터의 항목들을 그룹으로 간주하고 항목들에 관한 기술 통계량으로 탐색
#### aggregate : 그룹 간의 기술통계량
aggregate(mpg ~ cyl, data = mtcars, FUN = mean) # cyl에 따른 mpg의 평균 계산

boxplot(mpg ~ cyl, data = mtcars, main = "기통별 연비")


#-------------------------------------------------------------------------------


# 2. 상관 분석


# 2-2 상관관계의 표현 방법
library(mlbench)
data(PimaIndiansDiabetes)
df_pima <- PimaIndiansDiabetes[c(3:5, 8)]
str(df_pima)

summary(df_pima)

cor(df_pima, method = 'pearson')

cor(df_pima, method = 'spearman')

cor(df_pima, method = "kendall")

## 상관계수 시각화 - corrplot
windows(width = 12, height = 10)

library('corrplot')

corrplot(cor(df_pima), method = 'circle', type = 'lower')

## 정규성 만족 여부 검정(귀무가설 : 정규분포 따른다)
shapiro.test(df_pima$triceps) 

shapiro.test(df_pima$insulin)

cor.test(df_pima$triceps, df_pima$insulin, method = "kendall")


#-------------------------------------------------------------------------------


# 3. 변수 선택

# 3-3 변수 선택 기법
## 3-3-1 변수 선택 방법
data(mtcars)

m1 <- lm(hp ~., data = mtcars)

m2 <- step(m1, direction = "both")

formula(m2)


## 3-3-2 파생변수 생성
### 파생변수 : 주어진 독립변수에서 분석 목적에 맞도록 파생해낸 변수
pima <- PimaIndiansDiabetes
summary(pima$age)

library(dplyr)
pima <- pima %>% mutate(age_gen = cut(pima$age, c(20, 40, 60, 100),
                                      right = FALSE, label = c("Young", "Middle", "Old")))
table(pima$age_gen)


## 3-3-3 더미 변수(범주 개수 - 1) 생성
### 회귀분석에서 범주형 변수의 각 범주를 0과 1의 값만으로 표현하여 생성한 변수
### 실질적으로 이산형 변수이지만 연속형 변수로 간주
### lm함수에서 범주형 변수는 자동으로 더미 변수로 변환하여 분석 수행
중요도 <- c('상', '중', '하')
df <- data.frame(중요도)
df 
transform(df,
          변수1 = ifelse(중요도 == "중", 1, 0),
          변수2 = ifelse(중요도 == "하", 1, 0))
