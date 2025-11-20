---
tags: Python, MLOps, FastAPI, Docker, Airflow, React, Monitoring
date: 2025
icon: 🎬
---

# MLOps Project: 영화 추천 시스템 개발

## 프로젝트 개요

데이터 수집부터 모델 학습, 서빙, 모니터링까지 머신러닝 모델의 전체 수명 주기를 관리하는 MLOps 파이프라인을 구축하는 프로젝트입니다.

GitHub Repository: [https://github.com/jkim1209/mlops-project](https://github.com/jkim1209/mlops-project)

Presentation Slides: [Google Drive](https://drive.google.com/file/d/125RqKVFW9l_Nk6yo8OS24BwmA_vdXj8Z)

## 나의 역할과 기여도

프로젝트의 머신러닝 및 백엔드 파트를 총괄하여, 모델 개발부터 API 서빙 및 프론트엔드 연동까지 핵심적인 역할을 담당했습니다.

- 모델링 및 실험 관리: Numpy 기반의 경량 MLP 추천 모델을 구현하고, MLflow를 이용해 모든 실험 과정과 결과를 체계적으로 추적하고 관리했습니다.
- API 서버 구축: FastAPI를 사용하여 모델의 추론 결과를 실시간으로 제공하는 API 서버를 구축했습니다. `/predict`, `/latest-recommendations` 등 다양한 엔드포인트를 설계하고 구현했습니다.
- 시스템 배포: 학습된 모델을 서빙하는 API와 React로 개발된 프론트엔드를 연동하여 사용자가 실제로 서비스를 이용할 수 있도록 했으며, Docker를 이용해 배포 환경을 구축했습니다.

## 주요 기술 및 구현 내용

### 사용 기술

- Python
- ML & Data Science: Numpy, Pandas, Scikit-learn
- MLOps & Backend: FastAPI, PostgreSQL, AWS S3, Docker, Airflow
- Frontend: React.js
- Monitoring: MLflow, Prometheus, Grafana, Loki

### 핵심 구현

- Numpy 기반 MLP 추천 모델 구현
- MLflow를 활용한 모델 학습 모니터링
- FastAPI 기반 실시간 모델 추론 API 서버
- Airflow DAG를 이용한 데이터-학습-추론 파이프라인 자동화
- Prometheus, Grafana, Loki를 활용한 실시간 서버 및 애플리케이션 모니터링 시스템 구축

### MLOps 아키텍처

![MLOps 아키텍처](/projects/assets/images/06/01.png)

## Troubleshooting

### 문제: 데이터베이스 적재 실패로 인한 추천 결과 누락

**문제 상황**

React 프론트엔드에서 추천 영화의 포스터와 제목이 표시되지 않는 문제가 발생했습니다. 원인 추적 결과, 사용자의 시청 기록이 PostgreSQL 데이터베이스에 전혀 저장되지 않고 있었음을 발견했습니다.

**해결 과정**

1. 데이터 적재 로직을 디버깅한 결과, **`release_date` 필드가 빈 문자열("")인 경우** SQL 쿼리문에서 에러가 발생하며 데이터 삽입 전체가 실패하고 있었습니다.
2. **문제 탐지 지연 원인**:. 문제 탐지 지연 원인: 모델 학습/추론 파이프라인은 DB가 아닌 로컬에 캐싱된 로그 파일을 참조하고 있어 정상 동작하는 것처럼 보였습니다. 이로 인해 데이터 적재단의 오류가 한동안 탐지되지 못했습니다.
3. `release_date` 필드가 빈 문자열일 경우를 **예외 처리(NULL 값으로 변환 등)** 하도록 데이터 적재 코드를 수정하여 문제를 해결했습니다.

![모니터링 대시보드](/projects/assets/images/06/02.png)

## 성과 및 결과

데이터 수집, 모델 학습, 배치 추론, API 서빙, 모니터링을 포함하는 완전한 MLOps 파이프라인을 성공적으로 구축 및 배포했습니다.

### 경험 및 교훈

- 개별 기술(Docker, FastAPI, Airflow 등)을 유기적으로 연결하여 하나의 완성된 서비스를 만드는 개발 경험을 쌓았습니다.
- 파이프라인의 한 부분에서 발생한 조용한 실패(silent failure)가 시스템 전체에 미치는 영향을 직접 경험하며, End-to-End 테스트와 견고한 로깅 시스템의 중요성을 깊이 체감했습니다.
- 익숙하지 않은 분야의 팀원들과 협업하며 서로의 전문성을 이해하고 조율하는 과정에서 협업 커뮤니케이션의 가치를 배웠습니다.

![최종 결과](/projects/assets/images/06/03.png)
