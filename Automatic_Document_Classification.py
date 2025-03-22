import os
import json
import time
import boto3
import requests
import tempfile
from urllib.parse import urlparse
import datetime
import re
from PyPDF2 import PdfReader

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
    def __init__(self, sqs_queue_url=None, api_endpoint=None):
        # SQS 설정
        self.sqs_queue_url = sqs_queue_url
        self.sqs_client = get_sqs_client() if sqs_queue_url else None
        
        # API 엔드포인트 설정
        self.api_endpoint = api_endpoint
    
    def download_file_from_s3(self, bucket, key):
        """S3에서 파일 다운로드"""
        try:
            # S3 클라이언트가 없으면 생성
            if not hasattr(self, 's3_client'):
                self.s3_client = boto3.client('s3')
                
            # 임시 파일로 다운로드
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[1], delete=False) as temp_file:
                self.s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=key,
                    Fileobj=temp_file
                )
                temp_path = temp_file.name
                
            return temp_path
        except Exception as e:
            print(f"S3 파일 다운로드 오류: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """PDF 파일에서 텍스트 추출 (간소화)"""
        try:
            # 기본 PyPDF2를 사용한 텍스트 추출 시도
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # None 체크
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"PDF 파일 '{pdf_path}' 텍스트 추출 오류: {e}")
            return ""
    
    def simple_classify_pdf(self, pdf_path):
        """간단한 PDF 분류"""
        # 기본 문서 유형
        document_types = ['자기소개서', '이력서', '학습자료', '레포트', '기타']
        
        # 간단한 키워드 목록
        keywords = {
            '자기소개서': ['지원동기', '자기소개', '성장과정', '역량', '목표', 'personal statement'],
            '이력서': ['학력', '경력', '자격증', '기술', '연락처', 'resume', 'CV'],
            '학습자료': ['학습', '교육', '교재', '강의', '예제', 'textbook', 'lesson'],
            '레포트': ['서론', '본론', '결론', '연구', '참고문헌', 'references', 'conclusion'],
            '기타': []
        }
        
        # 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return '기타'
        
        # 간단한 규칙 기반 분류
        scores = {doc_type: 0 for doc_type in document_types}
        text_lower = text.lower()
        
        # 키워드 점수 계산
        for doc_type, kw_list in keywords.items():
            for keyword in kw_list:
                if keyword.lower() in text_lower:
                    scores[doc_type] += 1
        
        # 최고 점수 문서 유형 반환
        max_score = max(scores.values())
        if max_score == 0:
            return '기타'
        
        for doc_type in document_types:
            if scores[doc_type] == max_score:
                return doc_type
                
    def advanced_classify_pdf(self, pdf_path):
        """향상된 PDF 분류 메소드"""
        # 기본 문서 유형
        document_types = ['자기소개서', '이력서', '학습자료', '레포트', '기타']
        
        # 각 문서 유형별 주요 키워드 정의 (확장)
        keywords = {
            '자기소개서': [
                # 한국어 키워드 (가중치 높음)
                ('지원동기', 5), ('자기소개', 5), ('성장과정', 4), ('장단점', 4), ('지원이유', 5), 
                ('역량', 3), ('목표', 2), ('비전', 2), ('성취', 2), ('자소서', 5), ('지원원', 4),
                ('열정', 2), ('관심분야', 3), ('포부', 3), ('지원자', 2), ('인재상', 3),
                # 영어 키워드
                ('personal statement', 5), ('statement of purpose', 5), ('motivation letter', 5), 
                ('cover letter', 5), ('self introduction', 5), ('background', 2), 
                ('achievements', 3), ('strengths', 3), ('weaknesses', 3), ('career goal', 3), 
                ('objective', 2), ('vision', 2), ('why I want', 4), ('reason for applying', 5)
            ],
            '이력서': [
                # 한국어 키워드
                ('학력', 5), ('경력', 5), ('자격증', 5), ('기술', 4), ('연락처', 5), ('프로젝트', 4), 
                ('업무', 4), ('성과', 3), ('경험', 4), ('담당업무', 5), ('인적사항', 5),
                ('생년월일', 5), ('주소', 3), ('핸드폰', 3), ('이메일', 3), ('전화번호', 3),
                ('최종학력', 5), ('졸업', 4), ('학교', 3), ('전공', 3), ('회사명', 5),
                # 영어 키워드
                ('resume', 5), ('curriculum vitae', 5), ('CV', 5), ('education', 5), 
                ('experience', 5), ('certificate', 4), ('skills', 4), ('contact', 5), 
                ('project', 3), ('work history', 5), ('responsibilities', 4),
                ('personal information', 5), ('employment history', 5)
            ],
            '학습자료': [
                # 한국어 키워드
                ('학습', 4), ('교육', 4), ('교재', 5), ('강의', 5), ('이론', 4), ('개념', 4), 
                ('설명', 3), ('예제', 5), ('연습문제', 5), ('참고', 2), ('실습', 4), 
                ('교과서', 5), ('문제', 4), ('구하시오', 5), ('구해라', 5), ('풀이', 5),
                ('정의', 3), ('공식', 4), ('방정식', 4), ('증명', 4), ('그래프', 3),
                # 영어 키워드
                ('lecture', 5), ('course material', 5), ('textbook', 5), ('learning', 4), 
                ('education', 3), ('theory', 4), ('concept', 3), ('example', 4), 
                ('exercise', 5), ('practice', 4), ('study guide', 5), ('lesson', 5),
                ('worksheet', 5), ('tutorial', 5), ('quiz', 5), ('homework', 5)
            ],
            '레포트': [
                # 한국어 키워드
                ('서론', 5), ('본론', 5), ('결론', 5), ('연구', 5), ('분석', 4), ('조사', 4), 
                ('참고문헌', 5), ('인용', 5), ('논의', 4), ('평가', 3), ('보고서', 5),
                ('논문', 5), ('초록', 5), ('요약', 3), ('실험', 4), ('통계', 3),
                ('표', 2), ('그림', 2), ('목차', 3), ('가설', 4), ('데이터', 3),
                # 영어 키워드
                ('introduction', 5), ('body', 3), ('conclusion', 5), ('research', 5), 
                ('analysis', 4), ('investigation', 4), ('references', 5), ('bibliography', 5), 
                ('citation', 5), ('discussion', 4), ('evaluation', 3), ('report', 5),
                ('abstract', 5), ('methodology', 5), ('findings', 4), ('results', 4)
            ],
            '기타': []  # 다른 카테고리에 해당하지 않는 경우
        }
        
        # 패턴 정의 (정규표현식)
        patterns = {
            '자기소개서': [
                r'(저는|제가).{0,50}(지원|희망)',  # 한국어 자기소개 패턴
                r'(I am|I\'m|I have).{0,50}(applying|interested in)'  # 영어 자기소개 패턴
            ],
            '이력서': [
                r'(생년월일|연락처|이메일|Date of Birth|DOB|Phone|Contact|Email).{0,30}:',  # 개인정보 패턴
                r'\d{4}\s*[년~-]\s*\d{1,2}\s*[월~-]',  # 날짜 패턴 (경력, 학력)
                r'(education|experience|skills|qualification)[\s\n]*:'  # 영어 이력서 섹션
            ],
            '학습자료': [
                r'(chapter|단원|목차|학습목표|contents|table of contents|learning objectives|lesson)',  # 학습자료 구조
                r'(문제\s*\d+|\d+\.\s*문제|Exercise\s*\d+|Problem\s*\d+)'  # 문제 번호 패턴
            ],
            '레포트': [
                r'(참고문헌|reference|인용|출처|bibliography|cited|source|abstract)',  # 레포트 요소
                r'(introduction|methodology|results|conclusion|discussion)',  # 학술 레포트 구조
                r'\[\d+\]',  # 인용 표기 패턴
                r'\(\w+,\s*\d{4}\)'  # 인용 표기 패턴 (저자, 연도)
            ]
        }
        
        # 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 10:  # 텍스트가 너무 적으면
            print(f"파일 '{pdf_path}'에서 충분한 텍스트를 추출할 수 없습니다.")
            return '기타'
        
        # 점수 초기화
        scores = {doc_type: 0 for doc_type in document_types}
        text_lower = text.lower()
        
        # 1. 키워드 점수 계산 (가중치 적용)
        for doc_type, kw_list in keywords.items():
            if doc_type == '기타':
                continue
                
            for keyword, weight in kw_list:
                # 키워드 등장 횟수 확인
                count = text_lower.count(keyword.lower())
                if count > 0:
                    # 가중치를 고려하여 점수 추가
                    scores[doc_type] += count * weight
        
        # 2. 정규표현식 패턴 매칭
        for doc_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # 패턴 매치는 높은 점수 부여
                    scores[doc_type] += len(matches) * 10
        
        # 3. 문서 구조 분석
        # 자기소개서 특성 (자신을 나타내는 말이 많음)
        first_person_pronouns = ['저는', '제가', '저의', '제', 'I ', 'my', 'me', 'mine']
        first_person_count = sum(text_lower.count(pronoun.lower()) for pronoun in first_person_pronouns)
        if first_person_count > 10:
            scores['자기소개서'] += first_person_count * 2
        
        # 이력서 특성 (글머리 기호나 행머리 기호가 많음)
        bullet_point_pattern = r'(•|\*|\-|\d+\.|\d+\))'
        bullet_points = re.findall(bullet_point_pattern, text)
        if len(bullet_points) > 5:
            scores['이력서'] += len(bullet_points) * 3
        
        # 학습자료 특성 (여러 수식이나 기호가 있음)
        equations = re.findall(r'[=\+\-\*\/\^\(\)\[\]\{\}]+', text)
        if len(equations) > 10:
            scores['학습자료'] += len(equations) * 2
        
        # 레포트 특성 (긴 문단이 많음)
        long_paragraphs = len([p for p in text.split('\n\n') if len(p) > 200])
        if long_paragraphs > 3:
            scores['레포트'] += long_paragraphs * 5
        
        # 최고 점수 문서 유형 반환
        max_score = max(scores.values())
        if max_score == 0:
            return '기타'
        
        # 디버깅을 위한 점수 출력
        print("문서 분류 점수:")
        for doc_type, score in scores.items():
            print(f"  {doc_type}: {score}")
        
        # 최고 점수 유형 반환 (동점인 경우 우선순위가 높은 유형 선택)
        for doc_type in document_types:
            if scores[doc_type] == max_score:
                return doc_type
    def send_category_to_api(self, request_id, category):
        """카테고리 분류 결과를 API로 전송"""
        if not self.api_endpoint:
            print("API 엔드포인트가 설정되지 않았습니다.")
            return False
        
        try:
            # API 응답 데이터 구성 (요구하는 형식으로)
            payload = {
                "request_id": request_id,
                "is_completed": True,
                "predicted_category": category
            }
            
            # API 엔드포인트로 결과 전송
            url = f"{self.api_endpoint}/ai-proxy/category-recommendation-results/{request_id}"
            print(f"API 요청 URL: {url}")
            print(f"API 요청 데이터: {payload}")
            
            response = requests.post(
                url,
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
        """SQS 메시지 처리 - 카테고리 분류"""
        # 메시지 본문 파싱 (더 엄격한 JSON 파싱)
        try:
            message_body = json.loads(message['Body'].replace("'", '"'))
        except json.JSONDecodeError as json_err:
            print(f"JSON 파싱 오류: {json_err}")
            print(f"원본 메시지: {message['Body']}")
            return False
        
        # 요청 유형 처리
        request_type = message_body.get('request_type')
        print(f"수신된 request_type: {request_type}")
        
        # 지원되는 요청 유형 확장
        supported_types = [
            'file_duplicate_check_embedding_file', 
            'category_recommendation'
        ]
        
        if request_type not in supported_types:
            print(f"지원하지 않는 request_type: {request_type}")
            return False
        
        # 공통 필드 확인
        request_id = message_body.get('request_id')
        if not request_id:
            print("메시지에 request_id가 없습니다.")
            return False
        
        # 요청 유형별 처리
        if request_type == 'file_duplicate_check_embedding_file':
            # 기존 S3 파일 처리 로직
            payload = message_body.get('payload', {})
            s3_bucket = payload.get('s3_bucket')
            s3_key = payload.get('s3_key')
            
            if not s3_bucket or not s3_key:
                print("메시지 페이로드에 s3_bucket 또는 s3_key가 없습니다.")
                return False
            
            print(f"파일 처리 중: {s3_bucket}/{s3_key}, 요청 ID: {request_id}")
            
            # S3에서 파일 다운로드
            temp_path = self.download_file_from_s3(s3_bucket, s3_key)
            if not temp_path:
                print("S3에서 파일 다운로드에 실패했습니다.")
                return False
            
            try:
                # PDF 파일 분류 (향상된 방식)
                category = self.advanced_classify_pdf(temp_path)    
                
                # 분석 결과를 API로 전송
                api_result = self.send_category_to_api(request_id, category)
                
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
        
        elif request_type == 'category_recommendation':
            payload = message_body.get('payload', {})
            s3_bucket = payload.get('s3_bucket')
            s3_key = payload.get('s3_key')
            
            if not s3_bucket or not s3_key:
                print("메시지 페이로드에 s3_bucket 또는 s3_key가 없습니다.")
                return False
            
            print(f"파일 처리 중: {s3_bucket}/{s3_key}, 요청 ID: {request_id}")
            
            # S3에서 파일 다운로드
            temp_path = self.download_file_from_s3(s3_bucket, s3_key)
            if not temp_path:
                print("S3에서 파일 다운로드에 실패했습니다.")
                return False
            
            try:
                # PDF 파일 분류 (향상된 방식)
                category = self.advanced_classify_pdf(temp_path)    
                
                # 분석 결과를 API로 전송
                api_result = self.send_category_to_api(request_id, category)
                
                # 임시 파일 삭제
                os.unlink(temp_path)
                
                return api_result
                
            except Exception as e:
                print(f"메시지 처리 오류: {e}")
                import traceback
                traceback.print_exc()
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


def main():
    """
    PDF 처리기 메인 함수 (MongoDB 없이 실행)
    """
    # 환경 변수 확인 (시스템에 이미 설정되어 있다고 가정)
    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("경고: AWS 자격 증명이 환경 변수에 설정되어 있지 않습니다.")
        print("AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 설정하세요.")
        print("설정 방법 (Windows CMD): set AWS_ACCESS_KEY_ID=your_key")
        print("설정 방법 (PowerShell): $env:AWS_ACCESS_KEY_ID='your_key'")
        print("설정 방법 (Linux/Mac): export AWS_ACCESS_KEY_ID=your_key")
        return
    
    # AWS 리전 설정 (환경 변수에 없으면 기본값 사용)
    if not os.environ.get('AWS_REGION'):
        os.environ['AWS_REGION'] = 'ap-northeast-2'
    
    # SQS 큐 URL 설정
    sqs_queue_url = "https://sqs.ap-northeast-2.amazonaws.com/864981757354/XRPedia-AI-Requests.fifo"
    
    # API 엔드포인트 설정 - 실제 엔드포인트로 변경 필요
    api_endpoint = "https://5erhg0u08g.execute-api.ap-northeast-2.amazonaws.com"
    
    # 폴링 간격 설정 (초)
    polling_interval = 1
    
    print(f"PDF 처리기 초기화 중...")
    print(f"SQS 큐 URL: {sqs_queue_url}")
    print(f"API 엔드포인트: {api_endpoint}")
    print(f"폴링 간격: {polling_interval}초")
    
    try:
        # PDF 처리기 초기화 (MongoDB 없음)
        processor = PDFProcessor(
            sqs_queue_url=sqs_queue_url,
            api_endpoint=api_endpoint
        )
        
        print("SQS 메시지 폴링 시작...")
        # 폴링 루프 시작 (Ctrl+C로 중지할 때까지 실행)
        processor.start_polling_loop(polling_interval=polling_interval)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()