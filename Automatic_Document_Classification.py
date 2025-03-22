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
        """간단한 PDF 분류 (MongoDB 없이)"""
        # 기본 문서 유형
        document_types = ['자기소개서', '이력서', '학습자료', '레포트', '기타']
        
        # 간단한 키워드 목록 (축소)
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
    
    def send_to_api(self, file_id, classification_result, original_url, request_id):
        """분류 결과를 API로 전송"""
        if not self.api_endpoint:
            print("API 엔드포인트가 설정되지 않았습니다.")
            return False
        
        try:
            payload = {
                "fileId": file_id,
                "classificationType": classification_result,
                "originalUrl": original_url,
                "requestId": request_id,
                "processedTimestamp": datetime.datetime.now().isoformat()
            }
            
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
        """SQS 메시지 처리 (MongoDB 없이)"""
        try:
            # 메시지 본문 파싱
            message_body = json.loads(message['Body'])

            # 파일 URL과 requestId 추출
            file_url = message_body.get('fileUrl')
            request_id = message_body.get('requestId')

            if not file_url:
                print("메시지에 fileUrl이 없습니다.")
                return False

            if not request_id:
                print("메시지에 requestId가 없습니다.")
                return False
                
            print(f"파일 URL 처리 중: {file_url}, 요청 ID: {request_id}")
            
            # URL에서 파일 다운로드
            temp_path, filename = self.download_file_from_url(file_url)
            if not temp_path:
                print("파일 다운로드에 실패했습니다.")
                return False
            
            try:
                # PDF 파일 분류 (간소화된 방식)
                doc_type = self.simple_classify_pdf(temp_path)
                
                # 임시 파일 ID 생성 (MongoDB 없이)
                file_id = f"temp_{int(time.time())}_{request_id}"
                
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