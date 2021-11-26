# R : 통계 프로그래밍 언어인 S언어를 기반으로 만들어진 오픈 소스 프로그래밍 언어

# 1. 기본 문법

# 1-1 주석
## 한줄 주석 : # 주석 내용
## 다수 줄 주석
comment = "주석
          주석"

# 1-2 도움말
help(print)


# 1-3 연산자

## 1-3-1 산술 연산자
7%%4 #왼쪽 값을 오른쪽 값으로 나눈 나머지 계산
7%/%4 #왼쪽 값을 오른쪽 값으로 나눈 몫 계산산


## 1-3-3 논리 연산자
TRUE & FALSE
3 & 0 # 3은 TRUE, 0은 FALSE
TRUE|FALSE # 하나가 참이면 참, 그렇지 않으면 거짓
!TRUE # 반대값 출력


# 1-5 데이터 타입
## 1-5-3 데이터 기본 타입
### mode : 객체의 형식인 numeric, character, logical 중 하나의 값 출력
mode(1)
mode('a')
mode(TRUE)

### typeof : mode 함수를 사용했을 경우 numeric으로 출력되는 값 중에 정수형일 경우 interger, 실수형일 경우 double, 나머지는 동일
typeof(5) # 실수형으로 인식
typeof(5L) # R에서 정숫값을 나타내기 위해 정수 뒤에 L을 붙여야 함


## 1-5-4 데이터의 값
### NA: 데이터의 값이 없음(결측값)
a <- NA
is.na(a)

### NULL : 변수가 초기화 되지 않는 경우
### NAN : 수학적으로 계산할 수 없는 값
### INF : 무한대


# 1-6 객체
## 1-6-2 벡터
x <- c(1, 2, 3, 4)
y <- c('1', 2, '3') # 원소 중 하나라도 문자형이 있으면 모든 원소는 모두 문자형
z <- 5:10 # 5 6 7 8 9 10

### rep : 지정한 벡터를 반복 횟수만큼 반복
### seq(from, to, by) : from 부터 to까지 by만큼 증가하는 수열
x <- rep(c(1,2,3,), 3) #1 2 3 1 2 3 1 2 3
y <- seq(1, 10, 3) # 1 4 7 10

### 벡터[-n] : n번째 요소를 제외한 모든 요소 변환
x <- c(3:8)
x[-4] # 3 4 5 7 8


## 1-6-3 리스트
### (키, 값) 형태로 데이터를 저장하는 모든 객체를 담을 수 있는 데이터
### 문자형, 논리형 벡터, 데이터 프레임, 함수 저장 가능
list(name = 'soo', height = 90)
list(name = 'soo', height = c(2, 6))

### 리스트 안에 리스트를 중첩 저장
list(x = list(a = c(1,2,3)),
     y = list(b = c(1,2,3,4)))


## 1-6-4 행렬
### 한 가지 유형의 스칼라만 저장 가능
### matrix(data, nrow, ncol, byrow, dimnames
matrix(c(1:9), nrow = 3, dimnames = list(c("t1", "t2", "t3"),
                                         c("c1", "c2", "c3")))

### dim(x) : 차원 확인
### dim(x) <- c(m, n) : 행렬을 mxn로 변환
### t(x) : 전치행렬
### solve(x) : 역행렬
### x%*%y : x, y 행렬곱


## 1-6-5 데이터 프레임
### 벡터들의 집합
d <- data.frame(a = c(1,2,3,4),
                b = c(2,3,4,5),
                e = c('M', 'F', 'M', 'F'))

### stringsAsFactors : TRUE(factor형), FALSE(문자형)


## 1-6-6 배열
### array(data, dim, dimnames)
rn = c("1행", "2행")
cn = c("1열", "2열")
mn = c("1차원", "2차원")
array(1:8, dim = c(2,2,2), 
      dimnames = list(rn, cn, mn))


## 1-6-7 팩터
### factor(x, levels, labels, ordered)
factor("s", levels = c("s", "l"))

factor(c("a", "b", "c"), ordered = TRUE)


# 1-7 데이터 결합
## 1-7-3 merge
### 공통된 열을 하나 이상 가지고 있는 두 데이터 프레임에 대하여 기준이 되는 특정 컬럼의 값이 같은 행끼리 묶음
### all : TRUE(공통 값이 없는 행에 대해서는 NA로 채운 뒤 전체 행 병합)
### FALSE(공통 행만 병합)
### all.x, all.y : all.x = TRUE(x 데이터의 모든 행이 결과에 포함), 
### all.y = TRUE(y 데이터의 모든 행이 결과에 포함)
x <- data.frame(name = c("a", "b", "c"), math = c(1,2,3))
y <- data.frame(name = c("c", "b", "d"), english = c(4,5,6))

merge(x, y)
merge(x, y, all.x = TRUE)
merge(x, y, all = TRUE)


# 1-8 조건문
## 1-8-3 switch문(뒷부분에서 많이 사용)
### 조건에 따라 여러 개의 경로 중 하나를 선택하여 명령어 실행
course = "C"
switch(course,
       "A" = "brunch",
       "B" = "lunch",
       "dinner") #해당되는 유형이 없는 경우 실행


# 1-9 반복문
## 1-9-1 for문
for(i in 1:4) {
  print(i)
}


## 1-9-2 while문
i = 1
while (i <= 4) {
  print(i)
  i = i+1
}


## 1-9-3 repeat문
### 블록 안의 문장을 반복해서 수행하다가, 특정 상황에 종료할 때 사용
i = 1
repeat{
  print(i)
  if(i >= 2){
    break
  }
  i = i+1
}


## 1-9-4 루프 제어 명령어
### 무한 루프 방지
### break : 반복문을 중간에 탈출하기 위해 사용
for (i in 1:5) {
  print(i)
  if(i >= 2){
    break
  }
}

### next : 반복문에서 다음 반복으로 넘어갈 수 있도록 함
for (i in 1:5){
  if(i == 2){
    next #i의 값이 2일 경우 next를 만나서 다음 반복인 3 실행
  }
  print(i)
} # 2를 제외한 1, 3, 4, 5 출력


# 1-10 사용자 정의 함수
funcs_abs = function(x){
  if(x < 0){
    return(-x)
  }else{
    return(x)
  }
}

funcs_abs(-10.9)
funcs_abs(10.1)

## 반환 값이 없는 경우에 생략할 수 있으며 생략할 경우 마지막 실행 결과 반환
func_diff = function(x, y){
  print(x)
  print(y)
  print(x - y)
}

val = func_diff(9, 1) # 9 1 8
val # 8(최종 결과만 저장됨)


#-------------------------------------------------------------------------------


# 2. 시각화 함수

# 2-1 graphics 패키지
## plot(x, y, xlab = , ylab = , main = 그래프 제목, type = 산점도 출력 형태)
## hist(x, xlab , ylab , main , freq = TRUE(도수) / FALSE(상대 도수), breaks = 계급 구간 지정)

## barplot(x, xlab, ylab, main, names.arg = 각 막대에 사용할 문자열 벡터)
### x축 : 범주형 / y축 : 수치형
sales <- c(15, 23, 5, 20)
seasons <- c("1분기", "2분기", "3분기", "4분기")
df <- data.frame(sales, seasons)
barplot(sales ~ seasons, data = df)

### height에 벡터가 주어진 경우
h <- c(15, 23, 5, 20)
name <- c("1분기", "2분기", "3분기", "4분기")
barplot(h, names.arg = name)

### height에 행렬이 주어진 경우
h <- table(iris$Species)
name <- c("1분기", "2분기", "3분기", "4분기")
barplot(h, ylab = "수량", main = "종별 수량")

## pie(x, labels = 파이 조각 이름, density = 사선의 밀도, angle = 사선의 각도, main =)
p <- c(15, 23, 5, 20)
l <- c("1분기", "2분기", "3분기", "4분기")

pie(x = p, labels = l, 
    density = 50, # 원그래프를 지정한 수만큼 사선을 그어서 표시
    angle = 30 * 1:4) # 첫번째 파이 조각부터 30도씩 사선을 그을 각도 지정

## boxplot
boxplot(iris$Sepal.Length ~ iris$Species,
        notch = TRUE,
        xlab = "종별",
        ylab = "꽃받침 길이",
        main = "종별 꽃받침 길이 분포")


# 2-2 ggplot 패키지
## data : 시각화하려는 데이터
## aesthetics : 축의 스케일, 색상, 채우기 등 미학적/시각적 속성 의미
## geometrics : 데이터를 표현하는 도형

library(ggplot2)


## 2-2-3 ggplot 패키지 기본 문법
### ggplot(data, aes(x, y)) + geom_xx()


## 2-2-4 ggplot 패키지의 주요 함수
### geom_bar, geom_col, geom_point, geom_line, geom_boxplot 등

ggplot(diamonds, aes(color)) + #x축에 들어갈 컬럼명(diamond 객체 중 color) 
  geom_bar()

ggplot(sleep, aes(x = ID, y = extra)) + geom_point()

ggplot(Orange, aes(age, circumference)) + 
  geom_line(aes(color = Tree)) # 색깔을 Tree 속성별로 다르게 설정

ggplot(data = airquality, 
       aes(x = Month, 
           y = Temp, 
           group = Month) # Month 기준으로 집계
       ) + geom_boxplot()