# Orbit AI Server

Orbit 그룹웨어의 FastAPI 기반 AI 서버입니다.

사내 문서와 회의록 데이터를 기반으로 임베딩 검색을 수행하고,  
Qdrant Vector DB와 연동해 AI 챗봇의 비정형 문서 검색 기능을 담당합니다.

## 주요 기능

- 문서 / 회의록 검색 API
- Qdrant Vector DB 연동
- 임베딩 기반 유사도 검색
- Hybrid RAG 문서 검색 처리
- Spring Boot 백엔드와 API 연동
- Docker 기반 실행 환경 구성

## Tech Stack

| Category | Stack |
| --- | --- |
| Language | Python |
| Framework | FastAPI |
| Vector DB | Qdrant |
| Search | Vector Similarity Search |
| Architecture | Hybrid RAG |
| Infra | Docker |

## 역할

Spring Boot 백엔드에서 전달한 문서 검색 요청을 처리하고,  
Qdrant에 저장된 문서/회의록 임베딩 데이터 중 사용자 질문과 유사한 내용을 검색해 반환합니다.
