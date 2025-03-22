import numpy as np
import os
import hashlib
from sentence_transformers import SentenceTransformer
import torch
import re
from PyPDF2 import PdfReader
import docx
import pytesseract
from PIL import Image
import mimetypes
import faiss
from pymongo import MongoClient
from bson import ObjectId
import datetime
import boto3
import json
import requests
import tempfile
from urllib.parse import urlparse
import time

class DuplicateDetectionProcessor:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_gpu=False,
                 similarity_threshold=0.85, sqs_queue_url=None, api_endpoint=None,
                 region_name='ap-northeast-2'):
        """
        SQS와 API 통합을 위한 문서 중복 감지 프로세서 (MongoDB 없음)
        """
        # SQS 설정
        self.sqs_queue_url = sqs_queue_url
        self.sqs_client = boto3.client('sqs', region_name=region_name) if sqs_queue_url else None
        
        # S3 클라이언트 설정
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # API 엔드포인트 설정
        self.api_endpoint = api_endpoint
        
        # 모델 설정 (실제 사용하지 않음 - 간단한 구현을 위해)
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.similarity_threshold = similarity_threshold
    
    def download_file_from_s3(self, s3_url):
        """S3 URL에서 파일 다운로드"""
        try:
            # s3://bucket/key 형식에서 버킷과 키 추출
            if s3_url.startswith('s3://'):
                parts = s3_url[5:].split('/', 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ''
            else:
                raise ValueError(f"잘못된 S3 URL 형식: {s3_url}")
            
            # 파일명 추출
            filename = os.path.basename(key)
            if not filename:
                filename = f"file_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 임시 파일로 다운로드
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
                self.s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=key,
                    Fileobj=temp_file
                )
                temp_path = temp_file.name
            
            return temp_path, filename
        except Exception as e:
            print(f"S3 파일 다운로드 오류: {e}")
            return None, None
    
    def send_to_api(self, request_id, file_id, is_duplicated):
        """분석 결과를 API로 전송"""
        if not self.api_endpoint:
            print("API 엔드포인트가 설정되지 않았습니다.")
            return False
        
        try:
            # API 응답 데이터 구성
            result_data = {
                'request_id': request_id,
                'file_id': file_id,
                'is_completed': True,
                'is_duplicated': is_duplicated
            }
            
            # API 엔드포인트로 결과 전송
            url = f"{self.api_endpoint}/ai-proxy/file-duplicate-check-embeddings"
            print(f"API 요청 URL: {url}")
            print(f"API 요청 데이터: {result_data}")
            
            response = requests.post(
                url,
                json=result_data,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            print(f"API 전송 성공: {response.status_code}")
            return True
        
            
        except Exception as e:
            print(f"API 전송 오류: {e}")
            return False
    
    def process_sqs_message(self, message):
        """SQS 메시지 처리 (MongoDB 없이 간소화)"""
        try:
            # 메시지 본문 파싱
            message_body = json.loads(message['Body'])
            
            # 필수 필드 확인
            request_type = message_body.get('request_type')
            if request_type != 'file_duplicate_check_embedding_file':
                print(f"지원하지 않는 request_type: {request_type}")
                return False
            
            request_id = message_body.get('request_id')
            if not request_id:
                print("메시지에 request_id가 없습니다.")
                return False
            
            # 페이로드 확인
            payload = message_body.get('payload', {})
            s3_url = payload.get('s3_url')
            if not s3_url:
                print("메시지 페이로드에 s3_url이 없습니다.")
                return False
            
            user_id = message_body.get('user_id')
            
            print(f"파일 처리 중: {s3_url}, 요청 ID: {request_id}")
            
            # S3에서 파일 다운로드
            temp_path, filename = self.download_file_from_s3(s3_url)
            if not temp_path:
                print("S3에서 파일 다운로드에 실패했습니다.")
                return False
            
            try:
                # 실제로는 여기서 중복 분석을 수행하지만, 
                # MongoDB 없이 간단하게 구현하기 위해 항상 중복 아님으로 응답
                file_id = f"temp_{int(time.time())}"  # 임시 파일 ID 생성
                is_duplicated = False  # 중복 아님으로 설정
                
                # 분석 결과를 API로 전송
                api_result = self.send_to_api(request_id, file_id, is_duplicated)
                
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

class DuplicateDetectionModel:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_gpu=False, similarity_threshold=0.85,
                 mongo_uri='mongodb://localhost:27017', db_name='document_db'):
        """
        문서 중복 검사를 위한 머신러닝 모델 (MongoDB 통합)
        
        Args:
            model_name (str): 사용할 임베딩 모델 이름
            use_gpu (bool): GPU 사용 여부
            similarity_threshold (float): 중복으로 판단할 유사도 임계값 (0~1)
            mongo_uri (str): MongoDB 연결 URI
            db_name (str): 사용할 데이터베이스 이름
        """
        self.similarity_threshold = similarity_threshold
        
        # MongoDB 연결 설정
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.documents_collection = self.db['documents']
        self.embeddings_collection = self.db['embeddings']
        self.file_hashes_collection = self.db['file_hashes']
        
        # 임베딩 모델 로드
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
        
        # FAISS 인덱스 초기화
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.vector_dimension)  # 내적(코사인 유사도용)
        
        # 기존 임베딩 데이터 로드
        self._load_embeddings_from_mongodb()
    
    def _load_embeddings_from_mongodb(self):
        """MongoDB에서 임베딩 데이터 로드하여 FAISS 인덱스에 추가"""
        embeddings_docs = list(self.embeddings_collection.find({}))
        
        if len(embeddings_docs) > 0:
            # 문서 ID 목록 구성
            self.document_ids = [str(doc['document_id']) for doc in embeddings_docs]
            
            # 임베딩 벡터 로드하여 FAISS 인덱스에 추가
            embeddings = np.array([doc['embedding'] for doc in embeddings_docs], dtype=np.float32)
            if len(embeddings) > 0:
                self.index.add(embeddings)
                print(f"Loaded {len(embeddings)} embeddings from MongoDB")
        else:
            self.document_ids = []
            print("No embeddings found in MongoDB")
    
    def _extract_text(self, file_path):
        """파일에서 텍스트 추출"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type is None:
            # 확장자로 파일 타입 추정
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                mime_type = 'application/pdf'
            elif ext in ['.doc', '.docx']:
                mime_type = 'application/msword'
            elif ext in ['.txt', '.md']:
                mime_type = 'text/plain'
            elif ext in ['.jpg', '.jpeg', '.png']:
                mime_type = 'image/jpeg'
        
        # 파일 타입에 따라 텍스트 추출
        if mime_type and mime_type.startswith('text/'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
            
        elif mime_type == 'application/pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        elif mime_type and (mime_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml') or mime_type == 'application/msword'):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
            
        elif mime_type and mime_type.startswith('image/'):
            # 이미지에서 OCR로 텍스트 추출
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
            
        return ""
    
    def _compute_file_hash(self, file_path):
        """파일의 MD5 해시 계산"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # 64kb 단위로 읽기
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _preprocess_text(self, text):
        """텍스트 전처리"""
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _compute_embedding(self, text):
        """텍스트 임베딩 계산"""
        if not text:
            return np.zeros(self.vector_dimension)
        return self.model.encode(text, convert_to_numpy=True)
    
    def add_document(self, file_path, metadata=None):
        """
        문서 추가 및 중복 검사
        
        Args:
            file_path (str): 파일 경로
            metadata (dict, optional): 문서 관련 메타데이터
            
        Returns:
            dict: 중복 검사 결과
        """
        # 파일 해시 확인
        file_hash = self._compute_file_hash(file_path)
        
        # MongoDB에서 해시로 중복 확인
        existing_hash = self.file_hashes_collection.find_one({"hash": file_hash})
        if existing_hash:
            # 완전히 동일한 파일을 발견
            duplicate_doc = self.documents_collection.find_one({"_id": existing_hash["document_id"]})
            return {
                'is_duplicate': True,
                'duplicate_type': 'exact',
                'duplicate_id': str(existing_hash["document_id"]),
                'document_title': duplicate_doc.get('title', 'Unknown'),
                'similarity': 1.0
            }
        
        # 텍스트 추출 및 임베딩 계산
        text = self._extract_text(file_path)
        processed_text = self._preprocess_text(text)
        embedding = self._compute_embedding(processed_text)
        
        # 벡터 정규화 (코사인 유사도를 위해)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        embedding_for_faiss = normalized_embedding.reshape(1, -1).astype(np.float32)
        
        # 유사 문서 검색 (기존 문서가 있을 경우)
        similar_doc_info = None
        if len(self.document_ids) > 0:
            # FAISS로 유사도 검색
            D, I = self.index.search(embedding_for_faiss, min(5, len(self.document_ids)))
            
            # 유사도가 임계값을 넘는지 확인
            if D[0][0] > self.similarity_threshold:
                most_similar_idx = I[0][0]
                similar_doc_id = self.document_ids[most_similar_idx]
                
                # 유사 문서 정보 가져오기
                similar_doc = self.documents_collection.find_one({"_id": ObjectId(similar_doc_id)})
                
                similar_doc_info = {
                    'is_duplicate': True,
                    'duplicate_type': 'similar',
                    'duplicate_id': similar_doc_id,
                    'document_title': similar_doc.get('title', 'Unknown'),
                    'similarity': float(D[0][0])
                }
        
        # 중복이 아니거나, 사용자가 중복을 무시하고 추가하기를 원할 경우
        if similar_doc_info is None:
            # 파일명을 기본 제목으로 사용
            file_name = os.path.basename(file_path)
            
            # 문서 메타데이터 준비
            if metadata is None:
                metadata = {}
                
            # MongoDB에 문서 정보 저장
            document_data = {
                "title": metadata.get("title", file_name),
                "file_path": file_path,
                "file_type": mimetypes.guess_type(file_path)[0],
                "text_content": text[:1000],  # 전체 텍스트 저장하지 않고 일부만 저장 (필요시 변경)
                "created_at": datetime.datetime.now(),
                "metadata": metadata
            }
            
            document_id = self.documents_collection.insert_one(document_data).inserted_id
            
            # 파일 해시 저장
            self.file_hashes_collection.insert_one({
                "hash": file_hash,
                "document_id": document_id,
                "created_at": datetime.datetime.now()
            })
            
            # 임베딩 저장
            embedding_data = {
                "document_id": document_id,
                "embedding": normalized_embedding.tolist(),
                "created_at": datetime.datetime.now()
            }
            self.embeddings_collection.insert_one(embedding_data)
            
            # FAISS 인덱스 업데이트
            self.index.add(embedding_for_faiss)
            self.document_ids.append(str(document_id))
            
            return {
                'is_duplicate': False,
                'document_id': str(document_id),
                'title': document_data["title"]
            }
        
        return similar_doc_info
    
    # 나머지 클래스 메소드는 동일하게 유지...


def main():
    """
    중복 감지 프로세서 메인 함수 (MongoDB 없이 실행)
    """
    # SQS 큐 URL 설정
    sqs_queue_url = "https://sqs.ap-northeast-2.amazonaws.com/864981757354/XRPedia-AI-Requests.fifo"
    
    # API 엔드포인트 설정
    api_endpoint = "https://5erhg0u08g.execute-api.ap-northeast-2.amazonaws.com"
    
    # 기타 설정
    use_gpu = True  # 실제로는 사용하지 않음
    polling_interval = 1
    region_name = 'ap-northeast-2'
    
    print("중복 감지 프로세서 초기화 중...")
    print(f"SQS 큐 URL: {sqs_queue_url}")
    print(f"API 엔드포인트: {api_endpoint}")
    print(f"폴링 간격: {polling_interval}초")
    
    try:
        # MongoDB 없이 프로세서 초기화
        processor = DuplicateDetectionProcessor(
            sqs_queue_url=sqs_queue_url,
            api_endpoint=api_endpoint,
            region_name=region_name
        )
        
        print("SQS 메시지 폴링 시작...")
        processor.start_polling_loop(polling_interval=polling_interval)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()