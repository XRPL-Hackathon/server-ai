import os
import re
import json
import time
import boto3
import requests
import tempfile
from urllib.parse import urlparse
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

# MongoDB 연결 설정
def get_mongodb_connection(connection_string='mongodb://localhost:27017/'):
    """MongoDB 연결 객체 반환"""
    try:
        client = MongoClient(connection_string)
        # 연결 테스트
        client.admin.command('ping')
        print("MongoDB에 성공적으로 연결되었습니다.")
        return client
    except Exception as e:
        print(f"MongoDB 연결 오류: {e}")
        return None

# AWS SQS 설정
def get_sqs_client(region_name='ap-northeast-2'):
    """AWS SQS 클라이언트 반환"""
    try:
        # AWS 자격 증명을 환경 변수나 IAM 역할로부터 가져옴
        sqs = boto3.client('sqs', region_name=region_name)
        return sqs
    except Exception as e:
        print(f"AWS SQS 클라이언트 생성 오류: {e}")
        return None

class PDFProcessor:
    def __init__(self, mongodb_client=None, db_name="pdf_classifier_db", sqs_queue_url=None, api_endpoint=None):
        # MongoDB 설정
        self.mongodb_client = mongodb_client
        self.db_name = db_name
        
        if self.mongodb_client:
            self.db = self.mongodb_client[db_name]
            self.fs = gridfs.GridFS(self.db)
        else:
            self.db = None
            self.fs = None
        
        # SQS 설정
        self.sqs_queue_url = sqs_queue_url
        self.sqs_client = get_sqs_client() if sqs_queue_url else None
        
        # API 엔드포인트 설정
        self.api_endpoint = api_endpoint
        
        # PDF 분류기 초기화
        self.classifier = PDFClassifier(mongodb_client, db_name)
    
    def download_file_from_url(self, url):
        """URL에서 파일 다운로드"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 오류 확인
            
            # 파일명 추출
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            return temp_path, filename
        except Exception as e:
            print(f"파일 다운로드 오류: {e}")
            return None, None
    
    def send_to_api(self, file_id, classification_result, original_url, request_id):
        """분류 결과를 API로 전송"""
        if not self.api_endpoint:
            print("API 엔드포인트가 설정되지 않았습니다.")
            return False
        
        try:
            payload = {
                "fileId": str(file_id),
                "classificationType": classification_result,
                "originalUrl": original_url,
                "requestId": request_id,  # requestId 추가
                "processedTimestamp": pd.Timestamp.now().isoformat()
            }
            
            response = requests.post(
                f"{self.api_endpoint}/ai-proxy/category-recommendation-results/{request_id}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(f"API 전송 성공: {response.status_code}")
            return True
            
        except Exception as e:
            print(f"API 전송 오류: {e}")
            return False
    
    def process_sqs_message(self, message):
        """SQS 메시지 처리"""
        try:
            # 메시지 본문 파싱
            message_body = json.loads(message['Body'])

            # 파일 URL과 requestId 추출
            file_url = message_body.get('fileUrl')
            request_id = message_body.get('requestId')  # requestId 추출

            if not file_url:
                print("메시지에 fileUrl이 없습니다.")
                return False

            if not request_id:
                print("메시지에 requestId가 없습니다.")
                return False
            print(f"파일 URL 처리 중: {file_url}")
            
            # URL에서 파일 다운로드
            temp_path, filename = self.download_file_from_url(file_url)
            if not temp_path:
                print("파일 다운로드에 실패했습니다.")
                return False
            
            try:
                # MongoDB에 PDF 파일 업로드
                file_id = self.classifier.upload_pdf_to_db(temp_path, {"original_url": file_url})
                
                if not file_id:
                    print("MongoDB에 파일 업로드 실패")
                    return False
                
                # PDF 파일 분류
                doc_type = self.classifier.classify_pdf(temp_path, save_to_db=True)
                
                # 분류 결과를 API로 전송
                api_result = self.send_to_api(file_id, doc_type, file_url, request_id)
                
                # 임시 파일 삭제
                os.unlink(temp_path)
                
                return api_result
                
            except Exception as e:
                print(f"파일 처리 오류: {e}")
                # 임시 파일 삭제 시도
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return False
                
        except Exception as e:
            print(f"메시지 처리 오류: {e}")
            return False
    
    def poll_sqs_queue(self, max_messages=10, wait_time=20, visibility_timeout=60):
        """SQS 대기열에서 메시지 폴링"""
        if not self.sqs_client or not self.sqs_queue_url:
            print("SQS 클라이언트 또는 대기열 URL이 설정되지 않았습니다.")
            return False
        
        try:
            # SQS 대기열에서 메시지 수신
            response = self.sqs_client.receive_message(
                QueueUrl=self.sqs_queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
                VisibilityTimeout=visibility_timeout,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            processed_count = 0
            
            if not messages:
                print("처리할 메시지가 없습니다.")
                return True
            
            print(f"{len(messages)}개의 메시지를 처리합니다.")
            
            for message in messages:
                receipt_handle = message['ReceiptHandle']
                
                # 메시지 처리
                success = self.process_sqs_message(message)
                
                if success:
                    # 처리 성공 시 메시지 삭제
                    self.sqs_client.delete_message(
                        QueueUrl=self.sqs_queue_url,
                        ReceiptHandle=receipt_handle
                    )
                    processed_count += 1
                else:
                    # 처리 실패 시 메시지 가시성 타임아웃 변경 (다시 처리할 수 있도록)
                    self.sqs_client.change_message_visibility(
                        QueueUrl=self.sqs_queue_url,
                        ReceiptHandle=receipt_handle,
                        VisibilityTimeout=0  # 즉시 다시 처리 가능하도록
                    )
            
            print(f"{processed_count}/{len(messages)}개의 메시지를 성공적으로 처리했습니다.")
            return True
            
        except Exception as e:
            print(f"SQS 폴링 오류: {e}")
            return False
    
    def start_polling_loop(self, polling_interval=60):
        """SQS 대기열에서 메시지 폴링 루프 시작"""
        print(f"SQS 대기열 폴링 시작: {self.sqs_queue_url}")
        try:
            while True:
                self.poll_sqs_queue()
                # 다음 폴링 전 대기
                time.sleep(polling_interval)
        except KeyboardInterrupt:
            print("폴링 루프 중단")


# 기존 PDFClassifier 클래스 (필요한 메소드 추가)
class PDFClassifier:
    def __init__(self, mongodb_client=None, db_name="pdf_classifier_db"):
        # 문서 유형 정의
        self.document_types = ['자기소개서', '이력서', '학습자료', '레포트', '기타']
        
        # MongoDB 클라이언트 설정
        self.mongodb_client = mongodb_client
        self.db_name = db_name
        
        if self.mongodb_client:
            self.db = self.mongodb_client[db_name]
            self.fs = gridfs.GridFS(self.db)  # GridFS 설정 (PDF 파일 저장용)
        else:
            self.db = None
            self.fs = None
        
        # 각 문서 유형별 주요 키워드 정의
        self.keywords = {
            '자기소개서': [
                # 한국어 키워드
                '지원동기', '자기소개', '성장과정', '장단점', '지원이유', '역량', '목표', '비전', '성취', '자소서', '지원원',
                # 영어 키워드
                'personal statement', 'statement of purpose', 'motivation letter', 'cover letter', 
                'self introduction', 'background', 'achievements', 'strengths', 'weaknesses', 
                'career goal', 'objective', 'vision', 'why I want', 'reason for applying'
            ],
            '이력서': [
                # 한국어 키워드
                '학력', '경력', '자격증', '기술', '연락처', '프로젝트', '업무', '성과', '경험', '담당업무', '인적사항',
                # 영어 키워드
                'resume', 'curriculum vitae', 'CV', 'education', 'experience', 'certificate', 
                'skills', 'contact', 'project', 'work history', 'achievements', 'responsibilities',
                'personal information', 'employment history', 'qualification', 'professional experience'
            ],
            '학습자료': [
                # 한국어 키워드
                '학습', '교육', '교재', '강의', '이론', '개념', '설명', '예제', '연습문제', '참고', '실습', 
                '교과서', '문제', '구하시오' , '구해라',
                # 영어 키워드
                'lecture', 'course material', 'textbook', 'learning', 'education', 'theory', 
                'concept', 'example', 'exercise', 'practice', 'study guide', 'reference material',
                'handbook', 'tutorial', 'lesson', 'syllabus', 'curriculum', 'workbook', 'prove',
                'solve', 'proof'
            ],
            '레포트': [
                # 한국어 키워드
                '서론', '본론', '결론', '연구', '분석', '조사', '참고문헌', '인용', '논의', '평가', '보고서',
                # 영어 키워드
                'introduction', 'body', 'conclusion', 'research', 'analysis', 'investigation', 
                'references', 'bibliography', 'citation', 'discussion', 'evaluation', 'report',
                'methodology', 'finding', 'result', 'abstract', 'hypothesis', 'data'
            ],
            '기타': []  # 다른 카테고리에 해당하지 않는 경우
        }
        
        # 기계학습 모델 초기화
        self.model = None

    def upload_pdf_to_db(self, pdf_path, metadata=None):
        """PDF 파일을 MongoDB의 GridFS에 업로드"""
        if not self.db or not self.fs:
            print("MongoDB 연결이 설정되지 않아 업로드할 수 없습니다.")
            return None
            
        try:
            with open(pdf_path, 'rb') as pdf_file:
                file_data = pdf_file.read()
                
            # 기본 메타데이터 설정
            meta = {
                'filename': os.path.basename(pdf_path),
                'content_type': 'application/pdf',
                'uploaded_date': pd.Timestamp.now()
            }
            
            # 사용자 지정 메타데이터 추가
            if metadata:
                meta.update(metadata)
                
            # GridFS에 파일 저장
            file_id = self.fs.put(file_data, **meta)
            print(f"파일 '{os.path.basename(pdf_path)}'이(가) MongoDB에 업로드되었습니다. (ID: {file_id})")
            
            return file_id
            
        except Exception as e:
            print(f"PDF 업로드 오류: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """PDF 파일에서 텍스트 추출 (일반 + OCR)"""
        try:
            # 기본 PyPDF2를 사용한 텍스트 추출 시도
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # None 체크
                    text += page_text + "\n"
            
            # 추출된 텍스트가 충분하면 반환
            if text and len(text.strip()) > 100:
                return text
            
            # 텍스트가 충분하지 않으면 OCR 시도
            print(f"'{pdf_path}' 파일을 OCR로 처리합니다...")
            return self.extract_text_with_ocr(pdf_path)
            
        except Exception as e:
            print(f"PDF 파일 '{pdf_path}' 일반 추출 오류: {e}")
            # 일반 추출에 실패해도 OCR 시도
            print(f"'{pdf_path}' 파일을 OCR로 처리합니다...")
            return self.extract_text_with_ocr(pdf_path)
    
    def extract_text_with_ocr(self, pdf_path):
        """OCR을 사용한 텍스트 추출"""
        try:
            # pytesseract와 pdf2image를 사용한 OCR 처리
            from pdf2image import convert_from_path
            import pytesseract
            
            # PDF를 이미지로 변환
            images = convert_from_path(pdf_path)
            
            # 각 이미지에서 OCR로 텍스트 추출
            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='kor+eng')  # 한국어 + 영어 지원
                if page_text:
                    text += f"=== Page {i+1} ===\n{page_text}\n\n"
            
            return text
        except Exception as e:
            print(f"OCR 처리 오류: {e}")
            return ""
    
    def rule_based_classification(self, text):
        """규칙 기반 문서 분류"""
        scores = {doc_type: 0 for doc_type in self.document_types}
        
        # 각 문서 유형별 키워드 점수 계산
        for doc_type, keywords in self.keywords.items():
            for keyword in keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    scores[doc_type] += 1
        
        # 추가적인 규칙: 문서 구조 분석
        # 자기소개서 패턴 (한국어 + 영어)
        if re.search(r'(저는|제가).{0,50}(지원|희망)', text) or re.search(r'(I am|I\'m|I have).{0,50}(applying|interested in)', text, re.IGNORECASE):
            scores['자기소개서'] += 3
            
        # 이력서 패턴 (한국어 + 영어)
        if re.search(r'(생년월일|연락처|이메일|Date of Birth|DOB|Phone|Contact|Email).{0,30}:', text, re.IGNORECASE) or \
           re.search(r'\d{4}\s*[년~-]\s*\d{1,2}\s*[월~-]', text) or \
           re.search(r'(education|experience|skills|qualification)[\s\n]*:', text, re.IGNORECASE):
            scores['이력서'] += 3
            
        # 학습자료 패턴 (한국어 + 영어)
        if re.search(r'(chapter|단원|목차|학습목표|contents|table of contents|learning objectives|lesson)', text, re.IGNORECASE):
            scores['학습자료'] += 3
            
        # 레포트 패턴 (한국어 + 영어)
        if re.search(r'(참고문헌|reference|인용|출처|bibliography|cited|source|abstract)', text, re.IGNORECASE) or \
           re.search(r'(introduction|methodology|results|conclusion|discussion)', text, re.IGNORECASE):
            scores['레포트'] += 3
        
        # 가장 높은 점수의 문서 유형 반환 (동점일 경우 우선순위 적용)
        max_score = max(scores.values())
        if max_score == 0:
            return '기타'  # 모든 점수가 0이면 기타로 분류
        
        for doc_type in self.document_types:
            if scores[doc_type] == max_score:
                return doc_type
    
    def classify_pdf(self, pdf_path, use_model=False, save_to_db=True):
        """PDF 파일 분류하기"""
        text = self.extract_text_from_pdf(pdf_path)
        
        # 텍스트가 여전히 너무 적은 경우 AI 내용 분석 시도
        if not text or len(text.strip()) < 50:
            print(f"'{pdf_path}' 파일을 AI 모델을 사용하여 분석합니다...")
            doc_type = self.classify_with_ai(pdf_path)
            if doc_type:
                result = {
                    'filename': os.path.basename(pdf_path),
                    'path': pdf_path,
                    'type': doc_type,
                    'classification_method': 'AI',
                    'extracted_text': text,
                    'classified_at': pd.Timestamp.now()
                }
                
                if save_to_db and self.db:
                    self.save_classification_to_db(result)
                
                return doc_type
            return '분류 불가'
        
        if use_model and self.model is not None:
            doc_type = self.model_based_classification(text)
            classification_method = 'model'
        else:
            # 규칙 기반 분류 실행
            doc_type = self.rule_based_classification(text)
            classification_method = 'rule_based'
            
            # '기타' 카테고리로 분류된 경우 추가 검사
            if doc_type == '기타':
                # 각 카테고리 키워드 히트 수 확인
                keyword_hits = {
                    category: sum(1 for keyword in keywords if keyword.lower() in text.lower())
                    for category, keywords in self.keywords.items() if category != '기타'
                }
                
                # 키워드 히트가 전혀 없으면 AI 기반 분석 시도
                if sum(keyword_hits.values()) == 0:
                    print(f"'{pdf_path}' 파일을 AI 모델을 사용하여 추가 분석합니다...")
                    ai_doc_type = self.classify_with_ai(pdf_path)
                    if ai_doc_type:
                        doc_type = ai_doc_type
                        classification_method = 'AI'
                    
                    # AI 분석도 실패하면 '분류 불가'로 변경
                    if doc_type == '기타':
                        doc_type = '분류 불가'
        
        # 분류 결과를 MongoDB에 저장
        if save_to_db and self.db:
            result = {
                'filename': os.path.basename(pdf_path),
                'path': pdf_path,
                'type': doc_type,
                'classification_method': classification_method,
                'extracted_text': text[:1000],  # 텍스트 일부만 저장
                'classified_at': pd.Timestamp.now()
            }
            self.save_classification_to_db(result)
        
        return doc_type
        
    def save_classification_to_db(self, result):
        """MongoDB에 분류 결과 저장"""
        if not self.db:
            print("MongoDB 연결이 설정되지 않아 저장할 수 없습니다.")
            return None
        
        try:
            # MongoDB에 분류 결과 저장
            inserted_id = self.db.classification_results.insert_one(result).inserted_id
            print(f"파일 '{result['filename']}'의 분류 결과가 MongoDB에 저장되었습니다. (ID: {inserted_id})")
            
            return inserted_id
        
        except Exception as e:
            print(f"MongoDB 저장 오류: {e}")
            return None
            
    def classify_with_ai(self, pdf_path):
        """AI 모델을 사용한 PDF 문서 분류"""
        try:
            # 가능하면 텍스트 추출 다시 시도 (OCR 포함)
            text = self.extract_text_from_pdf(pdf_path)
            if not text or len(text.strip()) < 50:
                # 텍스트 추출 실패 시 파일 직접 분석 시도
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                
                # 문서 메타데이터 분석
                metadata = doc.metadata
                
                # 파일명에서 힌트 얻기
                filename = os.path.basename(pdf_path)
                
                # 이미지만 있는 PDF인지 확인
                has_images = False
                has_text = False
                
                for page in doc:
                    if page.get_text().strip():
                        has_text = True
                    if page.get_images():
                        has_images = True
                
                # 문서 특성으로 유형 추정
                if '이력서' in filename or '자기소개서' in filename:
                    return '이력서' if '이력서' in filename else '자기소개서'
                
                if metadata.get('title') and any(kw in metadata.get('title', '').lower() for kw in ['report', '보고서', 'analysis', '분석']):
                    return '레포트'
                
                if metadata.get('title') and any(kw in metadata.get('title', '').lower() for kw in ['교재', '학습', '강의', 'textbook', 'lecture']):
                    return '학습자료'
                
                # 외부 AI 서비스 호출 코드는 생략 (기존과 동일)
                
                # 실제 API 호출 없이 문서 특성 기반 추정 반환
                if not has_text and has_images:
                    # 이미지만 있는 경우 학습자료일 가능성 높음
                    return '학습자료'
                
            else:
                # 텍스트 있는 경우 내용 기반 추정
                text_lower = text.lower()
                
                # 각 문서 유형별 특성 점수 계산
                scores = {}
                
                for doc_type, keywords in self.keywords.items():
                    if doc_type == '기타':
                        continue
                    
                    # 키워드 등장 횟수로 점수 계산
                    score = sum(10 for kw in keywords if kw.lower() in text_lower)
                    
                    # 추가 패턴 검사
                    if doc_type == '자기소개서' and (
                        '저는' in text or '제가' in text or 'I am' in text_lower or 'my name is' in text_lower
                    ):
                        score += 20
                    
                    elif doc_type == '이력서' and (
                        re.search(r'\d{4}[-~년]\s*\d{1,2}[-~월]', text) or  # 날짜 패턴
                        re.search(r'(경력|학력|자격증|기술|experience|education|skills)', text_lower)
                    ):
                        score += 20
                    
                    elif doc_type == '학습자료' and (
                        re.search(r'(chapter|단원|목차|학습목표|예제|문제)', text_lower) or
                        text.count('\n\n') > 10  # 많은 문단 구분
                    ):
                        score += 20
                    
                    elif doc_type == '레포트' and (
                        re.search(r'(서론|본론|결론|참고문헌|introduction|conclusion|references)', text_lower) or
                        re.search(r'\[\d+\]', text)  # 인용 패턴
                    ):
                        score += 20
                    
                    scores[doc_type] = score
                
                # 가장 높은 점수 유형 반환
                if scores:
                    max_score = max(scores.values())
                    if max_score > 0:
                        for doc_type, score in scores.items():
                            if score == max_score:
                                return doc_type
            
            return None  # 분류 불가
            
        except Exception as e:
            print(f"AI 기반 분류 오류: {e}")
            return None


# 메인 함수 (AWS SQS + MongoDB 사용 예시)
def main():
    # MongoDB 연결
    mongo_client = get_mongodb_connection("mongodb://localhost:27017/")
    
    if not mongo_client:
        print("MongoDB 연결 실패. 프로그램을 종료합니다.")
        return
    
    # SQS 대기열 URL 및 API 엔드포인트 설정
    sqs_queue_url = "https://sqs.ap-northeast-2.amazonaws.com/your-account-id/your-queue-name"
    api_endpoint = "https://your-api-endpoint.com"
    
    # PDF 처리기 초기화
    processor = PDFProcessor(
        mongodb_client=mongo_client,
        db_name="pdf_classifier_db",
        sqs_queue_url=sqs_queue_url,
        api_endpoint=api_endpoint
    )
    
    # SQS 폴링 루프 시작
    processor.start_polling_loop()

if __name__ == "__main__":
    main()